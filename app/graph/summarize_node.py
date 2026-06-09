"""Memoria dinámica: resume la conversación cuando supera X tokens.

LangGraph (con SqliteSaver) persiste el estado, pero NO resume por sí solo:
ese resumen automático es este nodo. Al superar el umbral, condensa los
mensajes antiguos en ``summary`` y los elimina del historial, conservando los
últimos N intercambios.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from ..config import settings
from ..llm import get_chat_model
from .state import GraphState


def _approx_tokens(messages: list) -> int:
    chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    return chars // 4  # heurística ~4 chars/token, suficiente para el umbral


def summarize_node(state: GraphState) -> dict:
    messages = state["messages"]
    if _approx_tokens(messages) < settings.summary_token_threshold:
        return {}

    keep = settings.keep_last_messages
    to_summarize = messages[:-keep] if keep else messages
    if not to_summarize:
        return {}

    instruction = "Resume de forma concisa la siguiente conversación, conservando datos y decisiones clave."
    if state.get("summary"):
        instruction = (
            f"Resumen actual:\n{state['summary']}\n\n"
            "Amplíalo integrando los siguientes mensajes nuevos, de forma concisa."
        )

    convo = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)
    summary = get_chat_model(temperature=0.0).invoke(
        [
            SystemMessage(content="Eres un asistente que resume conversaciones."),
            HumanMessage(content=f"{instruction}\n\n{convo}"),
        ]
    ).content.strip()

    removals = [RemoveMessage(id=m.id) for m in to_summarize if getattr(m, "id", None)]
    return {"summary": summary, "messages": removals}
