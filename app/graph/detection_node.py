"""Nodo de detección de objetos: ejecución estática, sin pasar por el LLM.

El orquestador solo enruta aquí cuando el usuario pide explícitamente detectar
o contar personas/coches y hay una imagen disponible (subida en este turno o
ya presente en el contexto de la conversación, vía ``detect_image_path``).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from .state import GraphState
from .tools import detect_objects_in_image


def detection_node(state: GraphState) -> dict:
    image_path = state.get("detect_image_path")
    if not image_path:
        return {
            "messages": [AIMessage(content="No tengo ninguna imagen disponible para detectar objetos.")],
            "retrieved_image_paths": [],
        }

    result = detect_objects_in_image(image_path)
    if "error" in result:
        return {
            "messages": [AIMessage(content=result["error"])],
            "retrieved_image_paths": [],
        }

    counts = result["counts"]
    text = (
        f"He detectado {counts.get('person', 0)} persona(s) y "
        f"{counts.get('car', 0)} coche(s) en la imagen."
    )

    annotated_path = result.get("annotated_path")
    return {
        "messages": [AIMessage(content=text)],
        "retrieved_image_paths": [str(annotated_path)] if annotated_path else [],
    }
