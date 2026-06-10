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
    "- 'detect': el usuario pide detectar o contar personas/coches en una imagen (la subida "
    "en este turno o una mostrada antes en la conversación).\n"
    "- 'rag': preguntas sobre los documentos/base de conocimiento o sobre el contenido visual "
    "de una imagen subida (que no sea contar personas/coches).\n"
    "- 'python': el usuario pide cálculos, manipular datos, graficar o ejecutar código.\n"
    "- 'chitchat': saludos o charla general que no necesita documentos ni código.\n"
    "Devuelve solo la etiqueta de la ruta."
)

VALID_ROUTES = ("rag", "python", "chitchat", "detect")


class Route(BaseModel):
    route: Literal["rag", "python", "chitchat", "detect"] = Field(description="Ruta elegida")


def _last_image_in_context(state: GraphState) -> str | None:
    images = state.get("retrieved_image_paths") or []
    return images[-1] if images else None


def orchestrator_node(state: GraphState) -> dict:
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
        route = parsed.route
    else:
        # Fallback: el modelo (2B) a veces devuelve solo la etiqueta como texto
        # plano en lugar de JSON, incumpliendo el response_format solicitado.
        raw_text = str(result["raw"].content).strip().strip('"\'').lower()
        route = raw_text if raw_text in VALID_ROUTES else "chitchat"

    if route == "detect":
        # Solo se enruta a detección si hay una imagen disponible: la subida en
        # este turno o la última mostrada en el contexto de la conversación.
        image_path = state.get("image_path") or _last_image_in_context(state)
        if image_path:
            return {"route": "detect", "detect_image_path": image_path}
        # Pidió detección pero no hay ninguna imagen disponible.
        route = "chitchat"

    # Imagen subida en este turno sin petición de detección -> RAG (visión multimodal).
    if state.get("image_path"):
        return {"route": "rag"}

    return {"route": route}


def route_selector(state: GraphState) -> str:
    return state.get("route", "chitchat")
