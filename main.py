
import asyncio
import json
import logging
import random
import re
from contextlib import suppress
from datetime import datetime, date
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import discord
import edge_tts
import torch

from discord.ext import commands
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from config import DISCORD_TOKEN


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data"
AUDIO_DIR = BASE_DIR / "audio"
BIRTHDAYS_FILE = BASE_DIR / "birthdays.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_SIZE = 8 * 1024 * 1024
MAX_IMAGE_DIMENSION = 800

CONFIDENCE_THRESHOLD = 0.50
MAX_DISPLAYED_DETECTIONS = 15

DETECTOR_MODEL = "facebook/detr-resnet-50"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"

VOICE = "es-CO-SalomeNeural"

# Reducir el paralelismo de PyTorch para cuidar la CPU.
torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# REGISTRO DE EVENTOS Y ERRORES
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            BASE_DIR / "visionvox.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("VisionVox")


# ============================================================
# MODELOS Y CONTROL DE CONCURRENCIA
# ============================================================

detector = None
processor = None
captioner = None
translator_tokenizer = None
translator_model = None

models_ready = False

# Una sola inferencia de imagen a la vez.
scan_semaphore = asyncio.Semaphore(1)


# ============================================================
# INTENTS DE DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True


# ============================================================
# CARGA DE MODELOS
# ============================================================

def load_models():
    global detector
    global processor
    global captioner
    global translator_tokenizer
    global translator_model

    logger.info("Cargando detector DETR...")

    detector = pipeline(
        "object-detection",
        model=DETECTOR_MODEL,
        device=-1,
    )

    logger.info("Cargando modelo BLIP...")

    processor = BlipProcessor.from_pretrained(
        CAPTION_MODEL
    )

    captioner = BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL
    )
    captioner.eval()

    logger.info("Cargando traductor ingles-espanol...")

    translator_tokenizer = AutoTokenizer.from_pretrained(
        TRANSLATION_MODEL
    )

    translator_model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL
    )
    translator_model.eval()

    logger.info("Todos los modelos se cargaron correctamente.")


def models_are_ready():
    return all(
        model is not None
        for model in (
            detector,
            processor,
            captioner,
            translator_tokenizer,
            translator_model,
        )
    )


# ============================================================
# TRADUCCION CON RECUPERACION
# ============================================================

def translate_to_spanish(text):
    if not text or not text.strip():
        return ""

    try:
        inputs = translator_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.inference_mode():
            translated = translator_model.generate(
                **inputs,
                max_length=128,
                num_beams=1,
            )

        result = translator_tokenizer.decode(
            translated[0],
            skip_special_tokens=True,
        ).strip()

        return result or text

    except Exception:
        # Si falla la traduccion, conservar el texto original.
        logger.exception("Fallo la traduccion de un texto.")
        return text


# ============================================================
# DATOS CURIOSOS
# ============================================================

def get_fact():
    """Obtiene un dato curioso de la API y lo traduce al español."""
    url = "https://uselessfacts.jsph.pl/api/v2/facts/random"

    try:
        request = Request(
            url,
            headers={"User-Agent": "VisionVox Discord Bot"},
        )

        with urlopen(request, timeout=10) as response:
            if response.status != 200:
                return "No pude obtener un dato interesante."

            data = json.loads(response.read().decode("utf-8"))

        fact = data.get("text", "").strip()

        if not fact:
            return "No pude obtener un dato interesante."

        # Aprovecha el traductor Helsinki-NLP que VisionVox ya carga.
        translated_fact = translate_to_spanish(fact)

        return f"🧠 **¿Sabías que...?**\\n{translated_fact}"

    except (URLError, HTTPError, TimeoutError, OSError):
        logger.warning("No se pudo conectar con la API de datos curiosos.")
        return "⚠️ No pude conectarme a la API de datos curiosos."

    except Exception:
        logger.exception("Error al obtener o traducir el dato curioso.")
        return "⚠️ Ocurrió un error al obtener el dato curioso."


# ============================================================
# ANALISIS DE IMAGEN
# ============================================================

def analyze_image(image_path, annotated_path, report_progress):
    if not models_are_ready():
        raise RuntimeError(
            "Los modelos no estan disponibles."
        )

    with Image.open(image_path) as original:
        image = original.convert("RGB")

    image.thumbnail(
        (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
    )

    detections = []
    label_translations = {}
    warnings = []

    # --------------------------------------------------------
    # 1. DETECCION DE OBJETOS
    # --------------------------------------------------------

    report_progress("Detectando objetos con DETR...")

    try:
        raw_detections = detector(image)

        for item in raw_detections:
            confidence = float(item["score"])

            if confidence >= CONFIDENCE_THRESHOLD:
                box = item.get("box") or {}
                detections.append({
                    "label": item["label"],
                    "confidence": confidence,
                    "box": {
                        "xmin": float(box.get("xmin", 0)),
                        "ymin": float(box.get("ymin", 0)),
                        "xmax": float(box.get("xmax", 0)),
                        "ymax": float(box.get("ymax", 0)),
                    },
                })

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

    except Exception:
        logger.exception("Fallo la deteccion de objetos.")
        warnings.append(
            "No se pudieron detectar los objetos en esta imagen."
        )

    # --------------------------------------------------------
    # 2. TRADUCCION DE ETIQUETAS
    # --------------------------------------------------------

    report_progress("Preparando los nombres de los objetos...")

    for item in detections:
        label = item["label"]

        if label not in label_translations:
            label_translations[label] = translate_to_spanish(
                label
            )

    # --------------------------------------------------------
    # 3. DESCRIPCION DE ESCENA
    # --------------------------------------------------------

    report_progress("Generando la descripcion con BLIP...")

    caption = ""

    try:
        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = captioner.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
            )

        caption_en = processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

        report_progress("Traduciendo la descripcion...")

        caption = translate_to_spanish(caption_en)

        if not caption:
            caption = caption_en

    except Exception:
        logger.exception("Fallo la generacion de la descripcion.")
        warnings.append(
            "No se pudo generar la descripcion de la escena."
        )

    if not caption:
        caption = (
            "No hay una descripcion disponible para esta imagen."
        )

    # --------------------------------------------------------
    # 4. RESUMEN DE OBJETOS
    # --------------------------------------------------------

    unique_labels = list(dict.fromkeys(
        item["label"] for item in detections
    ))

    if unique_labels:
        names = [
            label_translations.get(label, label)
            for label in unique_labels
        ]

        object_summary = (
            "**Tipos de objetos detectados:** "
            + ", ".join(names)
        )
    else:
        object_summary = (
            "No se detectaron objetos con confianza igual o "
            f"superior al {CONFIDENCE_THRESHOLD * 100:.0f} %."
        )

    # --------------------------------------------------------
    # 4. IMAGEN ANOTADA CON LOS RECUADROS DE DETR
    # --------------------------------------------------------

    report_progress("Dibujando los objetos detectados en la imagen...")

    try:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()
        line_width = max(2, round(min(image.size) / 250))

        for item in detections[:MAX_DISPLAYED_DETECTIONS]:
            box = item.get("box", {})
            x1 = max(0, min(image.width, round(box.get("xmin", 0))))
            y1 = max(0, min(image.height, round(box.get("ymin", 0))))
            x2 = max(0, min(image.width, round(box.get("xmax", 0))))
            y2 = max(0, min(image.height, round(box.get("ymax", 0))))

            if x2 <= x1 or y2 <= y1:
                continue

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(255, 70, 70),
                width=line_width,
            )

            translated_label = label_translations.get(
                item["label"], item["label"]
            )
            text_label = (
                f"{translated_label} "
                f"{item['confidence'] * 100:.0f}%"
            )

            text_box = draw.textbbox((0, 0), text_label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_top = max(0, y1 - text_height - 8)

            draw.rectangle(
                (
                    x1,
                    label_top,
                    min(image.width, x1 + text_width + 8),
                    label_top + text_height + 6,
                ),
                fill=(255, 70, 70),
            )
            draw.text(
                (x1 + 4, label_top + 3),
                text_label,
                fill=(255, 255, 255),
                font=font,
            )

        annotated.save(annotated_path, format="PNG")

    except Exception:
        logger.exception("No se pudo crear la imagen anotada.")
        warnings.append(
            "No se pudo generar la imagen con los recuadros."
        )

    return {
        "detections": detections,
        "label_translations": label_translations,
        "annotated_path": str(annotated_path),
        "object_summary": object_summary,
        "caption": caption,
        "warnings": warnings,
    }


# ============================================================
# FORMATO DE RESULTADOS
# ============================================================

def format_results(results):
    lines = [
        "**VisionVox | Resultado del analisis**",
        "",
        results["object_summary"],
        "",
        (
            f"**Confianza minima:** "
            f"{CONFIDENCE_THRESHOLD * 100:.0f} %"
        ),
        "",
        "**Detecciones individuales:**",
    ]

    detections = results["detections"]

    if detections:
        for index, item in enumerate(
            detections[:MAX_DISPLAYED_DETECTIONS],
            start=1,
        ):
            label = results["label_translations"].get(
                item["label"],
                item["label"],
            )

            percent = item["confidence"] * 100

            lines.append(
                f"{index}. {label.capitalize()} — **{percent:.1f} %**"
            )

        remaining = len(detections) - MAX_DISPLAYED_DETECTIONS

        if remaining > 0:
            lines.append(
                f"... y {remaining} detecciones adicionales."
            )
    else:
        lines.append("No hay detecciones que mostrar.")

    lines.extend([
        "",
        "**Descripcion de la escena:**",
        results["caption"],
    ])

    if results["warnings"]:
        lines.extend([
            "",
            "**Avisos:**",
        ])

        for warning in results["warnings"]:
            lines.append(f"• {warning}")

    return "\n".join(lines)


async def send_long_message(ctx, content):
    """Dividir resultados largos respetando el limite de Discord."""

    limit = 1900
    remaining = content

    while remaining:
        if len(remaining) <= limit:
            chunk = remaining
            remaining = ""
        else:
            split_at = remaining.rfind("\n", 0, limit)

            if split_at <= 0:
                split_at = limit

            chunk = remaining[:split_at]
            remaining = remaining[split_at:].lstrip("\n")

        await ctx.send(chunk)


# ============================================================
# PROGRESO: SOLO CONSERVAR EL ESTADO MAS RECIENTE
# ============================================================

def report_progress_from_thread(loop, queue, status):
    """Enviar un estado desde un hilo sin bloquear PyTorch."""

    def update_queue():
        if queue.full():
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
                queue.task_done()

        queue.put_nowait(status)

    loop.call_soon_threadsafe(update_queue)


async def progress_worker(status_message, queue):
    while True:
        status = await queue.get()

        try:
            if status is None:
                return

            try:
                await status_message.edit(
                    content=f"🔄 {status}"
                )
            except discord.NotFound:
                logger.warning(
                    "El mensaje de progreso ya no existe."
                )
                return
            except discord.HTTPException:
                logger.warning(
                    "No se pudo actualizar el progreso."
                )

        finally:
            queue.task_done()


# ============================================================
# GENERACION DE AUDIO
# ============================================================

async def create_audio(text, audio_path):
    communicate = edge_tts.Communicate(
        text,
        VOICE,
    )

    await communicate.save(str(audio_path))


# ============================================================
# BOT DE DISCORD
# ============================================================

class VisionVoxBot(commands.Bot):

    async def setup_hook(self):
        global models_ready

        logger.info("Iniciando carga de modelos...")

        try:
            await asyncio.to_thread(load_models)

            if not models_are_ready():
                raise RuntimeError(
                    "La carga de modelos quedo incompleta."
                )

            models_ready = True
            logger.info("VisionVox esta listo.")

        except Exception:
            models_ready = False
            logger.exception(
                "No se pudo iniciar VisionVox correctamente."
            )
            raise


bot = VisionVoxBot(
    command_prefix="!",
    intents=intents,
    help_command=None,
)


@bot.event
async def on_ready():
    logger.info(
        "Conectado a Discord como %s",
        bot.user,
    )
    logger.info(
        "Estado de los modelos: %s",
        "listos" if models_ready else "no disponibles",
    )


# ============================================================
# COMANDOS
# ============================================================


@bot.command(name="help")
async def help_command(ctx):
    embed = discord.Embed(
        title="🤖 VisionVox | Ayuda",
        description="¡Estos son mis comandos disponibles!",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="🖼️ Análisis de imágenes",
        value="`!scan` — Analiza una imagen adjunta, detecta objetos, describe la escena y trata de generar audio.\n"
              "`!status` — Comprueba si los modelos de IA están listos.",
        inline=False,
    )
    embed.add_field(
        name="✨ Diversión",
        value="`!dato` — Dato curioso aleatorio traducido al español.\n"
              "`!bola <pregunta>` — La bola mágica responde.\n"
              "`!dado [caras]` — Lanza un dado (por defecto de 6 caras).\n"
              "`!hola` — Saludo aleatorio de VisionVox.",
        inline=False,
    )
    embed.add_field(
        name="⏰ Utilidades",
        value="`!timer <tiempo>` — Temporizador. Ejemplos: `!timer 30s`, `!timer 5m`, `!timer 1h` (máximo 2 horas).\n"
              "`!cumple DD-MM` — Registra tu cumpleaños (ejemplo: `!cumple 14-08`).\n"
              "`!cumples` — Muestra los cumpleaños registrados.",
        inline=False,
    )
    embed.add_field(
        name="ℹ️ Otros",
        value="`!hello` — Comprueba que el bot responde.\n"
              "`!visionhelp` — Ayuda alternativa.",
        inline=False,
    )
    embed.set_footer(text="VisionVox • ¡Que disfrutes el servidor!")
    await ctx.send(embed=embed)


@bot.command(name="hola")
@commands.cooldown(1, 3, commands.BucketType.user)
async def hola(ctx):
    greetings = [
        f"¡Hola, {ctx.author.mention}! 👋 ¿Qué tal?",
        f"¡Buenas, {ctx.author.mention}! VisionVox al habla 🤖",
        f"¡Hey, {ctx.author.mention}! ¡Qué bueno verte por aquí!",
        f"¡Holaaa, {ctx.author.mention}! ¿Listos para pasarla bien?",
    ]
    await ctx.send(random.choice(greetings))


@bot.command(name="bola")
@commands.cooldown(1, 3, commands.BucketType.user)
async def bola(ctx, *, question: str = None):
    if not question:
        await ctx.send("🔮 Escribe una pregunta. Ejemplo: `!bola ¿Tendremos suerte hoy?`")
        return

    answers = [
        "🔮 ¡Todo apunta a que sí!",
        "✨ ¡Parece que sí!",
        "🤔 No estoy seguro; pregunta de nuevo luego.",
        "🌧️ Mejor no contar con ello.",
        "🙃 Mi respuesta ahora mismo es que no.",
        "🍀 ¡Las posibilidades son buenas!",
        "🌀 El futuro está algo confuso...",
        "💫 ¡Definitivamente!",
        "📡 Mis circuitos dicen: puede ser.",
        "🚫 No parece probable.",
    ]
    await ctx.send(f"**Pregunta:** {question}\n{random.choice(answers)}")


@bot.command(name="dado")
@commands.cooldown(1, 2, commands.BucketType.user)
async def dado(ctx, sides: int = 6):
    if not 2 <= sides <= 100:
        await ctx.send("🎲 El dado puede tener entre 2 y 100 caras. Ejemplo: `!dado 20`")
        return
    result = random.randint(1, sides)
    await ctx.send(f"🎲 {ctx.author.mention} lanzó un dado de **{sides}** caras: **{result}**")


@bot.command(name="timer")
@commands.cooldown(1, 5, commands.BucketType.user)
async def timer(ctx, duration: str = None):
    if duration is None:
        await ctx.send("⏰ Indica el tiempo: `!timer 30s`, `!timer 5m` o `!timer 1h`.")
        return

    match = re.fullmatch(r"([1-9]\d*)(s|m|h)", duration.lower())
    if not match:
        await ctx.send("Formato inválido. Usa, por ejemplo, `30s`, `5m` o `1h`.")
        return

    amount = int(match.group(1))
    unit = match.group(2)
    seconds = amount * {"s": 1, "m": 60, "h": 3600}[unit]

    if seconds > 7200:
        await ctx.send("⏰ El temporizador máximo es de 2 horas.")
        return

    await ctx.send(f"⏳ Temporizador iniciado para **{duration}**.")
    await asyncio.sleep(seconds)
    with suppress(discord.HTTPException):
        await ctx.send(f"⏰ ¡Tiempo terminado, {ctx.author.mention}!")


def load_birthdays():
    try:
        if BIRTHDAYS_FILE.exists():
            content = json.loads(BIRTHDAYS_FILE.read_text(encoding="utf-8"))
            return content if isinstance(content, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.exception("No se pudo leer birthdays.json")
    return {}


def save_birthdays(birthdays):
    BIRTHDAYS_FILE.write_text(
        json.dumps(birthdays, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@bot.command(name="cumple")
async def cumple(ctx, birthday: str = None):
    if birthday is None:
        await ctx.send("🎂 Registra tu cumpleaños con `!cumple DD-MM`. Ejemplo: `!cumple 14-08`.")
        return

    match = re.fullmatch(r"(\d{1,2})-(\d{1,2})", birthday)
    if not match:
        await ctx.send("Formato incorrecto. Usa `DD-MM`, por ejemplo `!cumple 14-08`.")
        return

    day, month = int(match.group(1)), int(match.group(2))
    try:
        # Validar la fecha usando un año no bisiesto.
        date(2000, month, day)
    except ValueError:
        await ctx.send("Esa fecha no parece válida. Comprueba el día y el mes.")
        return

    birthdays = load_birthdays()
    birthdays[str(ctx.author.id)] = {
        "name": ctx.author.display_name,
        "date": f"{day:02d}-{month:02d}",
    }

    try:
        save_birthdays(birthdays)
    except OSError:
        logger.exception("No se pudo guardar un cumpleaños.")
        await ctx.send("⚠️ No pude guardar el cumpleaños en el archivo.")
        return

    await ctx.send(f"🎂 ¡Listo, {ctx.author.mention}! Guardé tu cumpleaños como **{day:02d}-{month:02d}**.")


@bot.command(name="cumples")
async def cumples(ctx):
    birthdays = load_birthdays()
    if not birthdays:
        await ctx.send("🎂 Todavía no hay cumpleaños registrados. Usa `!cumple DD-MM`.")
        return

    today = date.today()
    entries = []
    for user_id, item in birthdays.items():
        try:
            day, month = map(int, item["date"].split("-"))
            next_birthday = date(today.year, month, day)
            if next_birthday < today:
                next_birthday = date(today.year + 1, month, day)
            entries.append((next_birthday, item.get("name", "Usuario"), item["date"]))
        except (ValueError, KeyError, TypeError):
            continue

    entries.sort(key=lambda item: item[0])
    lines = [f"🎂 **{name}** — {birthday}" for _, name, birthday in entries[:25]]
    await ctx.send("**Cumpleaños registrados:**\n" + ("\n".join(lines) if lines else "No hay fechas válidas."))


@bot.command(name="hello")
async def hello(ctx):
    await ctx.send(
        "¡Hola! Soy VisionVox. Estoy listo para comenzar."
    )


@bot.command(name="visionhelp")
async def vision_help(ctx):
    await help_command(ctx)


@bot.command(name="dato")
@commands.cooldown(1, 5, commands.BucketType.user)
async def dato(ctx):
    await ctx.send("🧠 Buscando un dato curioso...")

    # Ejecutar la petición y traducción fuera del event loop.
    fact = await asyncio.to_thread(get_fact)
    await ctx.send(fact)


@bot.command(name="status")
async def status(ctx):
    if models_ready and models_are_ready():
        await ctx.send(
            "🟢 **VisionVox esta operativo.** "
            "Los modelos estan cargados y listos."
        )
    else:
        await ctx.send(
            "🟠 **VisionVox no esta listo para analizar.** "
            "Revisa la terminal."
        )


# ============================================================
# COMANDO SCAN
# ============================================================

@bot.command(name="scan")
@commands.cooldown(1, 5, commands.BucketType.user)
async def scan(ctx):
    image_path = None
    audio_path = None
    status_message = None
    progress_queue = None
    progress_task = None

    try:
        if not models_ready or not models_are_ready():
            await ctx.send(
                "⚠️ Los modelos todavia no estan disponibles. "
                "Intentalo de nuevo mas tarde."
            )
            return

        if not ctx.message.attachments:
            await ctx.send(
                "Adjunta una imagen y escribe `!scan` "
                "en el mismo mensaje."
            )
            return

        attachment = ctx.message.attachments[0]
        extension = Path(attachment.filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            await ctx.send(
                "Formato no admitido. Usa PNG, JPG, JPEG o WEBP."
            )
            return

        if attachment.size > MAX_IMAGE_SIZE:
            await ctx.send(
                "La imagen supera el limite de 8 MB."
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = (
            f"{ctx.author.id}_{timestamp}_{attachment.id}"
        )

        image_path = IMAGE_DIR / f"{unique_name}{extension}"
        annotated_path = IMAGE_DIR / f"{unique_name}_anotada.png"
        audio_path = AUDIO_DIR / f"{unique_name}.mp3"

        await ctx.send("📥 Imagen recibida. Validando archivo...")

        # Descargar y validar la imagen.
        try:
            await attachment.save(str(image_path))

            with Image.open(image_path) as test_image:
                test_image.verify()

        except (UnidentifiedImageError, OSError, ValueError):
            await ctx.send(
                "❌ El archivo no es una imagen valida "
                "o esta dañado."
            )
            return

        status_message = await ctx.send(
            "⏳ Imagen valida. Esperando turno de analisis..."
        )

        progress_queue = asyncio.Queue(maxsize=1)

        progress_task = asyncio.create_task(
            progress_worker(status_message, progress_queue)
        )

        loop = asyncio.get_running_loop()

        # Procesar una imagen a la vez.
        async with scan_semaphore:
            report_progress_from_thread(
                loop,
                progress_queue,
                "Iniciando analisis...",
            )

            results = await asyncio.to_thread(
                analyze_image,
                image_path,
                annotated_path,
                lambda message: report_progress_from_thread(
                    loop,
                    progress_queue,
                    message,
                ),
            )

        # Esperar a que se aplique el ultimo estado pendiente.
        await progress_queue.join()

        await status_message.edit(
            content="✅ Analisis terminado. Enviando resultados..."
        )

        await send_long_message(
            ctx,
            format_results(results),
        )

        # Enviar la imagen marcada, si se pudo generar.
        if annotated_path is not None and annotated_path.exists():
            try:
                await ctx.send(
                    "🖼️ **Imagen con los objetos detectados:**",
                    file=discord.File(
                        str(annotated_path),
                        filename="VisionVox_imagen_anotada.png",
                    ),
                )
            except discord.HTTPException:
                logger.exception("No se pudo enviar la imagen anotada.")
                await ctx.send(
                    "⚠️ El análisis de texto está disponible, "
                    "pero Discord no permitió enviar la imagen marcada."
                )

        # El audio es independiente del resultado de texto.
        try:
            await status_message.edit(
                content="🔊 Generando el audio de la descripcion..."
            )

            await create_audio(
                results["caption"],
                audio_path,
            )

            await ctx.send(
                "🎧 Audio de VisionVox:",
                file=discord.File(
                    str(audio_path),
                    filename="VisionVox_descripcion.mp3",
                ),
            )

            await status_message.edit(
                content="✅ Analisis completado."
            )

        except Exception:
            logger.exception("Fallo la generacion o envio de audio.")

            with suppress(discord.HTTPException):
                await status_message.edit(
                    content=(
                        "⚠️ El texto esta disponible, "
                        "pero no se pudo generar o enviar el audio."
                    )
                )

    except asyncio.CancelledError:
        logger.warning(
            "Analisis cancelado para el usuario %s.",
            ctx.author.id,
        )
        raise

    except discord.Forbidden:
        logger.exception(
            "Faltan permisos de Discord para completar la accion."
        )

    except discord.HTTPException:
        logger.exception("Error de comunicacion con Discord.")

        with suppress(discord.HTTPException):
            await ctx.send(
                "⚠️ Hubo un problema al comunicarse con Discord. "
                "Intentalo de nuevo mas tarde."
            )

    except Exception:
        logger.exception(
            "Fallo el comando scan para el usuario %s.",
            ctx.author.id,
        )

        if status_message is not None:
            with suppress(discord.HTTPException):
                await status_message.edit(
                    content=(
                        "❌ Ocurrio un error durante el analisis. "
                        "Revisa visionvox.log para mas detalles."
                    )
                )
        else:
            with suppress(discord.HTTPException):
                await ctx.send(
                    "❌ No pude completar el analisis. "
                    "Revisa visionvox.log para mas detalles."
                )

    finally:
        # Cancelar el actualizador sin dejar tareas pendientes.
        if progress_task is not None:
            progress_task.cancel()

            with suppress(asyncio.CancelledError):
                await progress_task

        # Eliminar solo los archivos temporales de esta solicitud.
        for temporary_path in (image_path, annotated_path, audio_path):
            if temporary_path is not None:
                try:
                    if temporary_path.exists():
                        temporary_path.unlink()
                except OSError:
                    logger.warning(
                        "No se pudo eliminar el archivo temporal: %s",
                        temporary_path,
                    )


# ============================================================
# ERRORES DE COMANDOS
# ============================================================

@bot.event
async def on_command_error(ctx, error):

    if isinstance(error, commands.CommandNotFound):
        return

    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(
            f"⏳ Espera {error.retry_after:.1f} segundos "
            "antes de volver a usar el comando."
        )
        return

    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(
            "Falta un argumento necesario para ese comando."
        )
        return

    logger.error(
        "Error de comando: %s",
        error,
        exc_info=(type(error), error, error.__traceback__),
    )

    with suppress(discord.HTTPException):
        await ctx.send(
            "⚠️ No pude completar el comando. "
            "Revisa visionvox.log para ver los detalles."
        )


# ============================================================
# INICIO
# ============================================================

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError(
            "No se encontro DISCORD_TOKEN. "
            "Comprueba tu archivo .env y config.py."
        )

    bot.run(DISCORD_TOKEN)
