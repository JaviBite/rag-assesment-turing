"""Tests del servicio de detección de objetos (Apartado 3).

Corre desde el HOST contra http://localhost:8002.
Uso: python tests/test_detector.py  (o make test-detector)

No necesita pytest: es ejecutable directamente para que sea fácil de correr
mientras el stack levanta.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

# Pillow puede no estar en el host; si no está, generamos los bytes a mano.
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DETECTOR_URL = os.getenv("DETECTOR_URL", "http://localhost:8002")


def _blank_png(width: int = 320, height: int = 240) -> bytes:
    if HAS_PIL:
        img = Image.new("RGB", (width, height), (200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    # PNG mínimo válido 1x1 en blanco (hardcodeado, sin deps).
    import zlib, struct  # noqa: E401
    def chunk(name, data):
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)
    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xFF\xFF\xFF"))
        + chunk(b"IEND", b"")
    )
    return raw


def _check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        sys.exit(1)


def test_health() -> None:
    print("\n-- health --")
    with urllib.request.urlopen(f"{DETECTOR_URL}/health", timeout=5) as r:
        body = json.loads(r.read())
    _check(body.get("status") == "ok", f"status == 'ok' (got {body})")


def test_detect_multipart_blank() -> None:
    """Imagen en blanco -> 0 detecciones, sin errores."""
    print("\n-- POST /detect (imagen en blanco) --")
    import urllib.request
    raw = _blank_png()
    boundary = b"----testboundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="blank.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
        + raw + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        f"{DETECTOR_URL}/detect",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        result = json.loads(r.read())
    _check("detections" in result, "response has 'detections' key")
    _check("counts" in result, "response has 'counts' key")
    _check("person" in result["counts"], "counts has 'person'")
    _check("car" in result["counts"], "counts has 'car'")
    # Solo las clases objetivo pueden aparecer.
    labels = {d["label"] for d in result["detections"]}
    _check(labels <= {"person", "car"}, f"only person/car in detections (got {labels})")
    print(f"  counts: {result['counts']}")


def test_detect_base64_blank() -> None:
    """Misma imagen en blanco pero enviada como base64."""
    print("\n-- POST /detect_base64 (imagen en blanco) --")
    raw = _blank_png()
    b64 = base64.b64encode(raw).decode()
    payload = json.dumps({"image_base64": b64}).encode()
    req = urllib.request.Request(
        f"{DETECTOR_URL}/detect_base64",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        result = json.loads(r.read())
    _check("detections" in result, "response has 'detections' key")
    print(f"  counts: {result['counts']}")


def test_detect_real_image_if_available() -> None:
    """Si hay una imagen real en docs/ o tests/, la usa para un test de humo."""
    candidates = list(Path(".").glob("tests/*.jpg")) + list(Path(".").glob("tests/*.png"))
    if not candidates:
        print("\n-- test imagen real: omitido (pon tests/*.jpg o tests/*.png) --")
        return
    img_path = candidates[0]
    print(f"\n-- POST /detect con {img_path.name} --")
    import urllib.request
    raw = img_path.read_bytes()
    boundary = b"----testboundary2"
    body = (
        b"--" + boundary + b"\r\n"
        + f'Content-Disposition: form-data; name="file"; filename="{img_path.name}"\r\n'.encode()
        + b"Content-Type: image/jpeg\r\n\r\n"
        + raw + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        f"{DETECTOR_URL}/detect",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read())
    labels = {d["label"] for d in result["detections"]}
    _check(labels <= {"person", "car"}, f"only person/car returned (got {labels})")
    print(f"  detecciones: {result['detections']}")
    print(f"  counts: {result['counts']}")


def test_invalid_image_returns_400() -> None:
    """Bytes aleatorios -> 400 Bad Request."""
    print("\n-- POST /detect con bytes inválidos (espera 400) --")
    raw = b"not an image at all"
    boundary = b"----testboundary3"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="bad.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
        + raw + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        f"{DETECTOR_URL}/detect",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            _check(False, "debería haber devuelto 400, no 200")
    except urllib.error.HTTPError as e:
        _check(e.code == 400, f"status code == 400 (got {e.code})")


if __name__ == "__main__":
    print(f"Detector URL: {DETECTOR_URL}")
    test_health()
    test_detect_multipart_blank()
    test_detect_base64_blank()
    test_detect_real_image_if_available()
    test_invalid_image_returns_400()
    print("\n✓ Todos los tests del detector pasaron.")
