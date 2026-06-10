"""Herramientas disponibles para los nodos del grafo."""
from __future__ import annotations

import base64
import uuid
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from ..config import settings

BOX_COLORS = {"person": (220, 60, 60), "car": (60, 140, 220)}
BOX_WIDTH = 3
FONT_SIZE = 15


def _draw_detections(image_path: Path, detections: list[dict]) -> Path | None:
    """Dibuja bounding boxes sobre la imagen original con Pillow y guarda el resultado."""
    if not detections:
        return None

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["box"]
        color = BOX_COLORS.get(label, (255, 200, 0))

        # Caja
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

        # Etiqueta con fondo de color para legibilidad
        text = f"{label} {conf:.0%}"
        try:
            tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1 - FONT_SIZE - 4), text, font=font)
        except AttributeError:
            # Pillow < 9.2
            tw, th = draw.textsize(text, font=font)
            tx1, ty1, tx2, ty2 = x1, y1 - th - 4, x1 + tw, y1
        draw.rectangle([tx1 - 2, ty1 - 2, tx2 + 2, ty2 + 2], fill=color)
        draw.text((tx1, ty1), text, fill=(255, 255, 255), font=font)

    settings.images_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.images_dir / f"detection_{uuid.uuid4().hex}.png"
    img.save(out_path, format="PNG")
    return out_path


def detect_objects_in_image(image_path: str) -> dict:
    """Llama al servicio de detección y dibuja las cajas sobre la imagen original.

    Devuelve un dict con ``counts`` y ``detections`` (del servicio) y
    ``annotated_path`` (ruta de la imagen con las cajas dibujadas, o None si no
    hubo detecciones). Si la imagen no existe, devuelve ``{"error": ...}``.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"No existe la imagen en {image_path}."}

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    resp = requests.post(
        f"{settings.detector_url}/detect_base64",
        json={"image_base64": b64},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    counts = data.get("counts", {})
    detections = data.get("detections", [])
    annotated_path = _draw_detections(path, detections)

    return {"counts": counts, "detections": detections, "annotated_path": annotated_path}
