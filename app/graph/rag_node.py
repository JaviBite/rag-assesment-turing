"""Nodo conversacional/RAG con acceso al tool de detección de objetos."""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..llm import get_chat_model, image_data_uri
from ..vectorstore import get_vectorstore
from .state import GraphState
from .tools import ANNOTATED_IMAGE_MARKER, detect_objects

TOOLS = [detect_objects]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

RAG_SYSTEM = (
    "Eres un asistente que responde basándose en los documentos indexados. "
    "Usa el CONTEXTO recuperado para responder y cita la fuente (documento y página). "
    "Si la respuesta no está en el contexto, dilo con claridad. "
    "Si el usuario pregunta cuántas personas o coches hay en una imagen subida, "
    "usa la herramienta detect_objects con la ruta de la imagen."
)


def _build_system_and_images(state: GraphState) -> tuple[SystemMessage, list[str]]:
    """Construye el system message con contexto RAG y devuelve las rutas de imágenes recuperadas."""
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
        parts.append("\nEl usuario ha adjuntado una imagen. Analízala visualmente si te lo pide.")

    return SystemMessage(content="\n".join(parts)), image_paths


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
    model = get_chat_model().bind_tools(TOOLS)
    system, image_paths = _build_system_and_images(state)
    messages = _inject_images_into_last_human(
        [system, *state["messages"]],
        user_image=state.get("image_path"),
        rag_images=image_paths,
    )

    response = model.invoke(messages)
    new_messages = [response]

    # Bucle de tool-calling (máximo 3 iteraciones para acotar el coste).
    for _ in range(3):
        if not getattr(response, "tool_calls", None):
            break
        messages.append(response)
        for call in response.tool_calls:
            tool = TOOLS_BY_NAME[call["name"]]
            result = tool.invoke(call["args"])
            tool_msg = ToolMessage(content=str(result), tool_call_id=call["id"])
            messages.append(tool_msg)
            new_messages.append(tool_msg)
            # Extraer ruta de imagen anotada si el tool la incluyó.
            for line in str(result).splitlines():
                if line.startswith(ANNOTATED_IMAGE_MARKER):
                    image_paths.append(line[len(ANNOTATED_IMAGE_MARKER):].strip())
        response = model.invoke(messages)
        new_messages.append(response)

    return {"messages": new_messages, "retrieved_image_paths": image_paths}


def chitchat_node(state: GraphState) -> dict:
    system_parts = ["Eres un asistente conversacional amable. Responde en español."]
    if state.get("summary"):
        system_parts.append(f"Resumen previo:\n{state['summary']}")
    messages = [SystemMessage(content="\n".join(system_parts)), *state["messages"]]
    response = get_chat_model().invoke(messages)
    return {"messages": [response], "retrieved_image_paths": []}
