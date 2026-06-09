"""Fixtures compartidos entre los tests que corren desde el host."""
import io
import os

import pytest
from PIL import Image, ImageDraw

DETECTOR_URL = os.getenv("DETECTOR_URL", "http://localhost:8002")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")


def make_blank_image(width: int = 320, height: int = 240) -> bytes:
    """Genera una imagen PNG en blanco para tests de smoke (sin detecciones)."""
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
