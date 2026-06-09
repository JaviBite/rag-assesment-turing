"""Interfaz de chatbot (Chainlit 2.x) sobre el grafo de LangGraph."""
from __future__ import annotations

import uuid
from pathlib import Path

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.graph import build_graph

GRAPH = build_graph()


@cl.on_chat_start
async def on_chat_start() -> None:
    # Un thread_id por sesión -> el checkpointer mantiene la memoria.
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(
        content=(
            "¡Hola! Soy tu asistente RAG. Puedes:\n"
            "- Preguntarme sobre los documentos indexados 📄\n"
            "- Subir una imagen para detectar personas y coches 🔍\n"
            "- Pedirme cálculos o código Python 🐍"
        )
    ).send()


def _save_uploaded_image(message: cl.Message) -> str | None:
    """Guarda la primera imagen adjunta en data/images/ y devuelve la ruta."""
    for element in message.elements or []:
        mime = getattr(element, "mime", "") or ""
        if not mime.startswith("image"):
            continue
        settings.images_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.images_dir / f"upload_{uuid.uuid4().hex}_{element.name}"
        if element.path:
            dest.write_bytes(Path(element.path).read_bytes())
        elif element.content:
            dest.write_bytes(
                element.content if isinstance(element.content, bytes)
                else element.content.encode()
            )
        else:
            continue
        return str(dest)
    return None


@cl.on_message
async def on_message(message: cl.Message) -> None:
    thread_id = cl.user_session.get("thread_id")
    image_path = _save_uploaded_image(message)

    config = {"configurable": {"thread_id": thread_id}}
    inputs: dict = {
        "messages": [HumanMessage(content=message.content)],
        "image_path": image_path,
    }

    result = await cl.make_async(GRAPH.invoke)(inputs, config=config)

    # Respuesta del asistente (último AIMessage con contenido).
    answer = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
        None,
    )

    # Imágenes recuperadas del RAG -> adjuntarlas como elementos inline.
    elements: list = []
    for img_path in result.get("retrieved_image_paths", []):
        p = Path(img_path)
        if p.exists():
            elements.append(
                cl.Image(name=p.name, path=str(p), display="inline")
            )

    await cl.Message(
        content=answer.content if answer else "_(sin respuesta)_",
        elements=elements,
    ).send()
