"""Heurísticas para detectar referencias a imágenes ya mostradas/recuperadas
en turnos anteriores de la conversación (p.ej. "describe la segunda imagen",
"la última foto", "la imagen número 3").

El router (modelo 2B) suele confundir estas peticiones de seguimiento con
nuevas consultas RAG, porque el propio prompt de la ruta 'rag' menciona
"imágenes... indexadas". Esta heurística se usa tanto para corregir el
enrutado (``orchestrator``) como para seleccionar la imagen a describir
(``chitchat_node``).
"""
from __future__ import annotations

import re

_IMAGE_WORD_RE = re.compile(r"\bimag(?:en|en[ée]s|ágenes)\b|\bfotos?\b", re.IGNORECASE)
_LAST_RE = re.compile(r"\b[uú]ltima?\b", re.IGNORECASE)
_NUMBER_AFTER_RE = re.compile(
    r"\bimag(?:en|en[ée]s|ágenes)\b(?:\s+n[uú]mero)?\s*(\d+)", re.IGNORECASE
)
_NUMBER_BEFORE_RE = re.compile(
    r"\b(\d+)\s*(?:ª|°|a)?\s*imag(?:en|en[ée]s|ágenes)\b", re.IGNORECASE
)

_ORDINALS: dict[str, int] = {
    "primera": 1, "primer": 1,
    "segunda": 2, "segundo": 2,
    "tercera": 3, "tercer": 3,
    "cuarta": 4, "cuarto": 4,
    "quinta": 5, "quinto": 5,
}


def select_referenced_image(text: str, images: list[str]) -> str | None:
    """Si ``text`` menciona una imagen/foto y hay ``images`` disponibles,
    devuelve la ruta referenciada (por posición ordinal/numérica, "última" o,
    por defecto, la última de la lista). Si ``text`` no menciona ninguna
    imagen, devuelve ``None`` (la heurística no aplica).
    """
    if not images or not _IMAGE_WORD_RE.search(text):
        return None

    text_low = text.lower()

    if _LAST_RE.search(text_low):
        return images[-1]

    match = _NUMBER_AFTER_RE.search(text_low) or _NUMBER_BEFORE_RE.search(text_low)
    if match:
        idx = int(match.group(1))
        if 1 <= idx <= len(images):
            return images[idx - 1]
        return images[-1]

    for word, idx in _ORDINALS.items():
        if idx <= len(images) and re.search(rf"\b{word}\b", text_low):
            return images[idx - 1]

    return images[-1]
