"""Test de la memoria dinámica con resumen automático.

Corre DENTRO del contenedor app:
    docker compose run --rm app python /srv/app/tests/test_memory.py

Pruebas:
1. El estado persiste entre turnos (SqliteSaver + thread_id).
2. El resumen se activa cuando los mensajes superan el umbral (se fuerza bajando el umbral).
3. Tras resumir, el contexto se mantiene coherente (el bot recuerda lo dicho antes).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/srv")

from langchain_core.messages import HumanMessage

from app.config import settings
from app.graph import build_graph


def _check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        sys.exit(1)


def invoke(graph, question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {"messages": [HumanMessage(content=question)], "image_path": None},
        config=config,
    )


def test_persistence(graph) -> None:
    """El bot recuerda el nombre mencionado en el turno anterior (mismo thread)."""
    print("\n-- Memoria: persistencia entre turnos --")
    tid = "test-mem-persistence"
    invoke(graph, "Me llamo Javier. Solo confirma que lo has anotado.", thread_id=tid)
    result = invoke(graph, "¿Cómo me llamo?", thread_id=tid)
    last = result["messages"][-1].content.lower()
    _check("javier" in last, f"el bot recuerda el nombre 'Javier' (got: {last[:200]})")
    print(f"  respuesta: {last[:200]}...")


def test_summary_triggered(graph) -> None:
    """Con un umbral muy bajo el nodo de resumen se activa."""
    print("\n-- Memoria: resumen automático al superar el umbral --")
    # Bajamos el umbral temporalmente para este test.
    original = settings.summary_token_threshold
    settings.summary_token_threshold = 50  # prácticamente siempre dispara
    tid = "test-mem-summary"

    try:
        for i in range(5):
            invoke(
                graph,
                f"Turno {i+1}: el número favorito de Javier es {42 + i}.",
                thread_id=tid,
            )
        result = invoke(graph, "¿De qué hemos hablado hasta ahora?", thread_id=tid)
        summary = result.get("summary", "")
        _check(len(summary) > 0, f"se generó un resumen (len={len(summary)})")
        print(f"  resumen generado ({len(summary)} chars): {summary[:200]}...")
    finally:
        settings.summary_token_threshold = original


def test_coherence_after_summary(graph) -> None:
    """Tras el resumen, el bot aún conoce datos mencionados antes del corte."""
    print("\n-- Memoria: coherencia tras resumen --")
    original = settings.summary_token_threshold
    settings.summary_token_threshold = 50
    tid = "test-mem-coherence"

    try:
        invoke(graph, "Mi ciudad favorita es Barcelona. Solo confirma.", thread_id=tid)
        for i in range(6):
            invoke(graph, f"Mensaje de relleno número {i+1} para forzar el resumen.", thread_id=tid)
        result = invoke(graph, "¿Cuál es mi ciudad favorita?", thread_id=tid)
        last = result["messages"][-1].content.lower()
        _check("barcelona" in last, f"ciudad 'Barcelona' sobrevive el resumen (got: {last[:200]})")
        print(f"  respuesta: {last[:200]}...")
    finally:
        settings.summary_token_threshold = original


if __name__ == "__main__":
    print("Construyendo el grafo...")
    graph = build_graph()
    test_persistence(graph)
    test_summary_triggered(graph)
    test_coherence_after_summary(graph)
    print("\n✓ Todos los tests de memoria pasaron.")
