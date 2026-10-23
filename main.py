
import asyncio
import logging
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import discord
import edge_tts
import torch

from discord.ext import commands
from PIL import Image, UnidentifiedImageError
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
# ANALISIS DE IMAGEN
# ============================================================

def analyze_image(image_path, report_progress):
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
                detections.append({
                    "label": item["label"],
                    "confidence": confidence,
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

    return {
        "detections": detections,
        "label_translations": label_translations,
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

@bot.command(name="hello")
async def hello(ctx):
    await ctx.send(
        "¡Hola! Soy VisionVox. Estoy listo para comenzar."
    )


@bot.command(name="visionhelp")
async def vision_help(ctx):
    await ctx.send(
        "**VisionVox - Comandos disponibles**\n"
        "`!hello` - Comprueba que el bot funciona.\n"
        "`!scan` - Analiza una imagen y genera un MP3.\n"
        "`!visionhelp` - Muestra esta ayuda."
    )


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
        for temporary_path in (image_path, audio_path):
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
