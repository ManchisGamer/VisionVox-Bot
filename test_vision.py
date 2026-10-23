
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
)


def main():
    print("Cargando detector de objetos...")

    detector = pipeline(
        "object-detection",
        model="facebook/detr-resnet-50",
        device=-1,
    )

    print("Cargando modelo de descripcion BLIP...")

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    captioner = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    captioner.eval()

    # Buscar imagenes en la carpeta data
    image_dir = Path("data")
    extensions = {".png", ".jpg", ".jpeg", ".webp"}

    if not image_dir.exists():
        print("No existe la carpeta data.")
        return

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]

    if not images:
        print("No hay imagenes en la carpeta data.")
        print("Envia una imagen al bot usando !scan.")
        return

    # Elegir la imagen guardada mas recientemente
    image_path = max(
        images,
        key=lambda path: path.stat().st_mtime,
    )

    print(f"\nAnalizando: {image_path.name}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as error:
        print(f"No se pudo abrir la imagen: {error}")
        return

    # Detectar objetos
    print("\nDetectando objetos...")

    objects = detector(image)

    if objects:
        for obj in objects:
            label = obj["label"]
            confidence = obj["score"] * 100
            print(f"- {label}: {confidence:.1f}%")
    else:
        print("No se detectaron objetos.")

    # Generar descripcion directamente con BLIP
    print("\nGenerando descripcion...")

    try:
        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = captioner.generate(
                **inputs,
                max_new_tokens=60,
            )

        description = processor.decode(
            output[0],
            skip_special_tokens=True,
        )

        print("\nDescripcion generada:")
        print(description)

    except Exception as error:
        print(f"Error al generar la descripcion: {error}")


if __name__ == "__main__":
    main()
