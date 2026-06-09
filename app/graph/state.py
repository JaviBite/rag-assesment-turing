"""Estado compartido del grafo."""
from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    # Historial de mensajes (reducer de LangGraph que acumula/actualiza).
    messages: Annotated[list, add_messages]
    # Resumen acumulado de la conversación antigua (memoria dinámica).
    summary: str
    # Rama elegida por el orquestador: "rag" | "python" | "chitchat".
    route: str
    # Ruta de una imagen subida por el usuario en el turno actual (o None).
    image_path: str | None
    # Rutas de imágenes recuperadas del RAG en el último turno (para mostrarlas en la UI).
    retrieved_image_paths: list[str]
