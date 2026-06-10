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
    # Rama elegida por el orquestador: "rag" | "python" | "chitchat" | "detect".
    route: str
    # Ruta de una imagen subida por el usuario en el turno actual (o None).
    image_path: str | None
    # Imágenes mostradas/usadas en el último turno (RAG y/o la subida por el
    # usuario): para mostrarlas en la UI y como "imagen en contexto" para
    # turnos siguientes (p.ej. detección de objetos sin volver a subirla).
    retrieved_image_paths: list[str]
    # Imagen objetivo para el nodo de detección: la subida en este turno o la
    # última disponible en el contexto de la conversación (o None).
    detect_image_path: str | None
