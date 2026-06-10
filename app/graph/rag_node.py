"""Nodo conversacional/RAG."""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_chat_model, image_data_uri
from ..vectorstore import get_vectorstore
from .state import GraphState

RAG_SYSTEM = (
    "Eres un asistente que responde basándose en los documentos indexados. "
    "Usa el CONTEXTO recuperado para responder y cita la fuente (documento y página). "
    "Si la respuesta no está en el contexto, dilo con claridad."
)


def _build_system_and_images(state: GraphState) -> tuple[SystemMessage, list[str], list]:
    """Construye el system message con contexto RAG y devuelve las rutas de imágenes y los docs recuperados."""
    parts = [RAG_SYSTEM]
    if state.get("summary"):
        parts.append(f"\nResumen de la conversación previa:\n{state['summary']}")

    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    query = str(last_human.content) if last_human else ""
    docs = get_vectorstore().similarity_search(query, k=4)

    image_paths: list[str] = []
    if docs:
        ctx_parts = []
        for d in docs:
            meta = d.metadata
            ctx_parts.append(
                f"[{meta.get('source')} p.{meta.get('page')} "
                f"({meta.get('type')})] {d.page_content}"
            )
            if meta.get("type") == "image" and meta.get("image_path"):
                image_paths.append(meta["image_path"])
        parts.append("\nCONTEXTO:\n" + "\n\n".join(ctx_parts))

    if state.get("image_path"):
        parts.append(
            f"\nEl usuario ha adjuntado una imagen en: {state['image_path']}. "
            "Analízala visualmente si te lo pide."
        )

    return SystemMessage(content="\n".join(parts)), image_paths, docs


def _format_sources_footer(docs: list) -> str:
    """Pie de "Fuentes:" a partir de los metadatos (documento + página) de los docs recuperados."""
    if not docs:
        return ""
    sources: dict[str, list[str]] = {}
    for d in docs:
        meta = d.metadata
        source = meta.get("source", "?")
        label = f"p. {meta.get('page', '?')}"
        if meta.get("type") == "image":
            label += " (imagen)"
        labels = sources.setdefault(source, [])
        if label not in labels:
            labels.append(label)
    lines = (f"- {src}: {', '.join(labels)}" for src, labels in sources.items())
    return "\n\n**Fuentes:**\n" + "\n".join(lines)


def _inject_images_into_last_human(
    messages: list, user_image: str | None, rag_images: list[str]
) -> list:
    """Sustituye el último HumanMessage por uno multimodal si hay imágenes que incluir."""
    all_images = [p for p in ([user_image] if user_image else []) + rag_images
                  if p and Path(p).exists()]
    if not all_images:
        return messages

    messages = list(messages)
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            text = messages[i].content if isinstance(messages[i].content, str) else ""
            content: list = [{"type": "text", "text": text}]
            for img_path in all_images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data_uri(img_path)},
                })
            messages[i] = HumanMessage(content=content)
            break
    return messages


def rag_node(state: GraphState) -> dict:
    system, image_paths, docs = _build_system_and_images(state)
    messages = _inject_images_into_last_human(
        [system, *state["messages"]],
        user_image=state.get("image_path"),
        rag_images=image_paths,
    )

    response = get_chat_model().invoke(messages)

    footer = _format_sources_footer(docs)
    if footer and isinstance(response.content, str):
        response.content += footer

    # Si el usuario adjuntó una imagen, queda "en contexto" para turnos siguientes
    # (p.ej. para pedir detección de objetos sobre ella sin volver a subirla).
    retrieved_images = list(image_paths)
    if state.get("image_path"):
        retrieved_images.append(state["image_path"])

    return {"messages": [response], "retrieved_image_paths": retrieved_images}


def chitchat_node(state: GraphState) -> dict:
    system_parts = ["Eres un asistente conversacional amable. Responde en español."]
    if state.get("summary"):
        system_parts.append(f"Resumen previo:\n{state['summary']}")
    messages = [SystemMessage(content="\n".join(system_parts)), *state["messages"]]
    response = get_chat_model().invoke(messages)
    return {"messages": [response], "retrieved_image_paths": []}
