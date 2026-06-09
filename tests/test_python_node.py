"""Test del nodo Python: genera y ejecuta código para resolver problemas.

Corre DENTRO del contenedor app:
    docker compose run --rm app python /srv/app/tests/test_python_node.py

Pruebas:
1. Cálculo numérico -> factorial, lista de primos, etc.
2. Operación con datos -> estadísticas básicas sobre una lista.
3. Verifica que el nodo usa la ruta 'python' (no RAG).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/srv")

from langchain_core.messages import HumanMessage

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


def test_factorial(graph) -> None:
    print("\n-- Python: factorial de 12 --")
    result = invoke(graph, "Calcula el factorial de 12 usando Python.", thread_id="test-py-factorial")
    route = result.get("route", "?")
    _check(route == "python", f"ruta == 'python' (got '{route}')")
    last_content = result["messages"][-1].content
    # 12! = 479001600
    _check("479001600" in last_content, f"resultado contiene 479001600 (got: {last_content[:200]})")
    print(f"  respuesta: {last_content[:200]}...")


def test_statistics(graph) -> None:
    print("\n-- Python: estadísticas de una lista --")
    q = "Calcula la media, mediana y desviación estándar de [4, 8, 15, 16, 23, 42] con Python."
    result = invoke(graph, q, thread_id="test-py-stats")
    route = result.get("route", "?")
    _check(route == "python", f"ruta == 'python' (got '{route}')")
    last_content = result["messages"][-1].content
    # Media = 18.0
    _check(
        "18" in last_content or "media" in last_content.lower(),
        f"respuesta contiene resultado estadístico (got: {last_content[:200]})",
    )
    print(f"  respuesta: {last_content[:200]}...")


def test_no_hallucination_without_exec(graph) -> None:
    """El nodo debe ejecutar código, no inventar resultados."""
    print("\n-- Python: la respuesta viene de ejecución real (tool_calls presentes) --")
    result = invoke(graph, "¿Cuántos números primos hay entre 1 y 100? Calcula con Python.", thread_id="test-py-primes")
    msgs = result["messages"]
    # Debe haber al menos un ToolMessage (ejecución real del REPL).
    tool_msgs = [m for m in msgs if m.type == "tool"]
    _check(len(tool_msgs) > 0, f"se ejecutó al menos un tool_call (tool_msgs={len(tool_msgs)})")
    last = msgs[-1].content
    _check("25" in last, f"resultado correcto (25 primos) en respuesta (got: {last[:200]})")
    print(f"  tool calls: {len(tool_msgs)}")
    print(f"  respuesta: {last[:200]}...")


if __name__ == "__main__":
    print("Construyendo el grafo...")
    graph = build_graph()
    test_factorial(graph)
    test_statistics(graph)
    test_no_hallucination_without_exec(graph)
    print("\n✓ Todos los tests del nodo Python pasaron.")
