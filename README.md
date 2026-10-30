# VisionVox 🔍🔊

**Un bot de Discord que utiliza inteligencia artificial para interpretar imágenes y convertir sus descripciones en audio en español.**

VisionVox es un proyecto desarrollado en Python que combina modelos de visión artificial y procesamiento de lenguaje natural para ayudar a interpretar el contenido de imágenes desde Discord.

El bot recibe una imagen, identifica objetos, genera una descripción de la escena, traduce los resultados al español y produce un archivo de audio.

## ✨ Características principales

- **Detección de objetos:** identifica objetos presentes en una imagen mediante DETR.
- **Porcentajes de confianza:** muestra el nivel de confianza de cada detección.
- **Recuadros visuales:** genera una imagen anotada con los objetos identificados y sus etiquetas.
- **Descripción de escenas:** utiliza BLIP para generar una descripción automática.
- **Traducción al español:** traduce las etiquetas y descripciones mediante Helsinki-NLP.
- **Conversión de texto a voz:** genera audio en español con Edge-TTS.
- **Integración con Discord:** permite realizar análisis mediante comandos.
- **Optimización para CPU:** limita el procesamiento paralelo para reducir el consumo de recursos.
- **Manejo de errores:** conserva los resultados de texto cuando falla la generación de audio.

## 🧠 Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lenguaje principal |
| discord.py | Integración con Discord |
| DETR | Detección de objetos |
| BLIP | Descripción de imágenes |
| Hugging Face Transformers | Ejecución de modelos de IA |
| Helsinki-NLP | Traducción inglés-español |
| PyTorch | Procesamiento de modelos |
| Pillow | Procesamiento y anotación de imágenes |
| Edge-TTS | Generación de voz |
| python-dotenv | Gestión de variables de entorno |

## 🏗️ Modelos de inteligencia artificial

VisionVox utiliza los siguientes modelos:

- **Detección:** `facebook/detr-resnet-50`
- **Descripción:** `Salesforce/blip-image-captioning-base`
- **Traducción:** `Helsinki-NLP/opus-mt-en-es`

Los modelos se descargan desde Hugging Face cuando se cargan por primera vez.

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ManchisGamer/VisionVox-Bot.git
```

Entrar en la carpeta:

```bash
cd VisionVox
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar el token de Discord

Crear un archivo llamado `.env` en la carpeta principal:

```env
DISCORD_TOKEN=TU_TOKEN_DE_DISCORD
```

**Importante:** nunca publiques tu token ni compartas el archivo `.env`.

### 5. Ejecutar VisionVox

```bash
python main.py
```

La primera ejecución puede tardar porque los modelos de inteligencia artificial deben descargarse y cargarse.

## 🤖 Comandos disponibles

| Comando | Descripción |
|---|---|
| `!hello` | Comprueba que el bot está funcionando. |
| `!scan` | Analiza una imagen adjunta y genera los resultados. |
| `!status` | Comprueba si los modelos están disponibles. |
| `!visionhelp` | Muestra la ayuda del bot. |

### Ejemplo de uso

1. Adjunta una imagen en un canal donde esté el bot.
2. Escribe `!scan` en el mismo mensaje.
3. Espera a que VisionVox procese la imagen.
4. Recibe las detecciones, la descripción, la imagen anotada y el audio cuando esté disponible.

## 🖥️ Requisitos

- Python compatible con las dependencias del proyecto.
- Conexión a Internet para Discord, la descarga inicial de modelos y Edge-TTS.
- Memoria RAM suficiente para ejecutar los modelos de visión artificial.
- Espacio de almacenamiento para las dependencias y los modelos descargados.

VisionVox está configurado para ejecutar los modelos mediante CPU (`device=-1`).

## 📁 Estructura del proyecto

```text
VisionVox/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── data/
│   └── Archivos temporales de imágenes
│
├── audio/
│   └── Archivos temporales de audio
│
└── visionvox.log
```

Los archivos `.env`, los registros y los archivos temporales no deben publicarse en GitHub.

## 🔐 Seguridad

- El token de Discord se obtiene mediante variables de entorno.
- El archivo `.env` debe mantenerse privado.
- Los archivos temporales se eliminan después del procesamiento.
- No se deben subir credenciales ni información privada al repositorio.

## 🚀 Estado del proyecto

VisionVox cuenta con las funciones principales de análisis de imágenes, descripción de escenas, traducción, generación de audio e integración con Discord.

El proyecto continúa en desarrollo y puede recibir nuevas mejoras.

## 👨‍💻 Autor

Desarrollado como proyecto de programación e inteligencia artificial.

---

**VisionVox — Dando voz a lo que vemos.** 🔍🔊
