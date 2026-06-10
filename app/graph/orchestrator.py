"""Nodo orquestador: clasifica la petición y decide la rama (routing explícito).

No se delega el routing al tool-choice del modelo (poco fiable en un 2B): se usa
salida estructurada para obtener una etiqueta y LangGraph enruta con edges.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..llm import get_chat_model
from .state import GraphState

ROUTER_SYSTEM = (
    "Eres un orquestador que clasifica la última petición del usuario en una de estas rutas:\n"
    "- 'rag': preguntas sobre los documentos/base de conocimiento, sus imágenes, o detectar "
    "objetos (personas/coches) en una imagen subida.\n"
    "- 'python': el usuario pide cálculos, manipular datos, graficar o ejecutar código.\n"
    "- 'chitchat': saludos o charla general que no necesita documentos ni código.\n"
    "Devuelve solo la etiqueta de la ruta."
)


class Route(BaseModel):
    route: Literal["rag", "python", "chitchat"] = Field(description="Ruta elegida")


def orchestrator_node(state: GraphState) -> dict:
    # Si hay imagen subida en este turno, va directo a RAG (tiene el tool de detección).
    if state.get("image_path"):
        return {"route": "rag"}

    last_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if last_user is None:
        return {"route": "chitchat"}

    model = get_chat_model(temperature=0.0).with_structured_output(
        Route, method="json_schema", include_raw=True
    )
    result = model.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=str(last_user.content)),
    ])

    parsed = result["parsed"]
    if parsed is not None:
        return {"route": parsed.route}

    # Fallback: el modelo (2B) a veces devuelve solo la etiqueta como texto
    # plano en lugar de JSON, incumpliendo el response_format solicitado.
    raw_text = str(result["raw"].content).strip().strip('"\'').lower()
    if raw_text in ("rag", "python", "chitchat"):
        return {"route": raw_text}
    return {"route": "chitchat"}


def route_selector(state: GraphState) -> str:
    return state.get("route", "chitchat")
