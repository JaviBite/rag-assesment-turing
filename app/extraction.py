"""Detección de formularios y extracción estructurada (Apartado 1, valorado +).

Estrategia simple: se renderiza la primera página, se pregunta a Gemma si es un
formulario y, en caso afirmativo, se extraen sus campos de forma estructurada.
El resultado se guarda como JSON en local (no en la vector DB, según el enunciado).
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .config import settings
from .llm import image_data_uri, get_chat_model


class _IsForm(BaseModel):
    is_form: bool = Field(description="True si la página es un formulario con campos a rellenar")


class FormField(BaseModel):
    field_name: str = Field(description="Nombre/etiqueta del campo del formulario")
    value: str | None = Field(default=None, description="Valor rellenado, o null si vacío")


class FormExtraction(BaseModel):
    form_type: str = Field(description="Tipo de formulario (p.ej. 'inscripción', 'solicitud')")
    nombre: str | None = Field(default=None, description="Valor del campo 'nombre', o null si vacío")
    apellidos: str | None = Field(default=None, description="Valor del campo 'apellidos', o null si vacío")
    fecha_nacimiento: str | None = Field(default=None, description="Valor del campo 'fecha de nacimiento', o null si vacío")
    fecha: str | None = Field(default=None, description="Valor del campo 'fecha', o null si vacío")
    lugar: str | None = Field(default=None, description="Valor del campo 'lugar', o null si vacío")
    dni: str | None = Field(default=None, description="Valor del campo 'DNI' o 'NIF', o null si vacío")                                  
    extra: list[FormField] = Field(description="Lista de campos extra relevantes detectados con su valor")


def is_form(image_path: str | Path) -> bool:
    """Pregunta binaria a Gemma sobre si la imagen es un formulario."""
    model = get_chat_model(temperature=0.0).with_structured_output(_IsForm, method="json_schema")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "¿Esta página es un formulario con campos a rellenar "
                    "(por ejemplo nombre, apellidos, fecha de nacimiento, casillas)?"
                ),
            },
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
    )
    return model.invoke([message]).is_form


def extract_form(image_path: str | Path) -> FormExtraction:
    """Extracción estructurada de los campos del formulario."""
    model = get_chat_model(temperature=0.0).with_structured_output(FormExtraction, method="json_schema")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Extrae todos los campos de este formulario con su valor. "
                    "Si un campo está vacío, devuelve null en su valor."
                ),
            },
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
    )
    return model.invoke([message])


def process_form(pdf_stem: str, image_path: Path) -> Path | None:
    """Si la página es un formulario, extrae y guarda el JSON. Devuelve la ruta o None."""
    if not is_form(image_path):
        return None
    extraction = extract_form(image_path)
    settings.extractions_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.extractions_dir / f"{pdf_stem}.json"
    out_path.write_text(
        json.dumps(extraction.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
