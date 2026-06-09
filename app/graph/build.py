"""Ensamblado del grafo multiagente con checkpointer persistente (SqliteSaver)."""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ..config import settings
from .orchestrator import orchestrator_node, route_selector
from .python_node import python_node
from .rag_node import chitchat_node, rag_node
from .state import GraphState
from .summarize_node import summarize_node


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("rag", rag_node)
    graph.add_node("python", python_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_selector,
        {"rag": "rag", "python": "python", "chitchat": "chitchat"},
    )
    # Todas las ramas pasan por el nodo de resumen antes de terminar.
    for branch in ("rag", "python", "chitchat"):
        graph.add_edge(branch, "summarize")
    graph.add_edge("summarize", END)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)
