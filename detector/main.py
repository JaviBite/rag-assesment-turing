"""Servicio local de detección de objetos (Apartado 3).

FastAPI + YOLO preentrenado. Recibe una imagen y devuelve un JSON con las
detecciones, filtrando exclusivamente las clases ``person`` y ``car``.
"""
from __future__ import annotations

import base64
import binascii
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("YOLO_CONF", "0.25"))
TARGET_CLASSES = {"person", "car"}

app = FastAPI(title="Object Detection Service", version="1.0.0")
model = YOLO(MODEL_PATH)


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    detections: list[Detection]
    counts: dict[str, int]


class Base64Request(BaseModel):
    image_base64: str


def _run_inference(image: Image.Image) -> DetectionResponse:
    """Ejecuta YOLO y filtra a las clases objetivo."""
    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)
    detections: list[Detection] = []
    counts = {cls: 0 for cls in TARGET_CLASSES}

    for result in results:
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls)]
            if label not in TARGET_CLASSES:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=round(float(box.conf), 4),
                    box=[round(float(v), 2) for v in box.xyxy[0].tolist()],
                )
            )
            counts[label] += 1

    return DetectionResponse(detections=detections, counts=counts)


def _load_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    """Detección a partir de fichero multipart (Postman, curl -F, etc.)."""
    raw = await file.read()
    return _run_inference(_load_image(raw))


@app.post("/detect_base64", response_model=DetectionResponse)
def detect_base64(payload: Base64Request) -> DetectionResponse:
    """Detección a partir de imagen en base64 (data-URI también admitida)."""
    data = payload.image_base64
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="base64 inválido") from exc
    return _run_inference(_load_image(raw))
