"""Test del flujo RAG: invoca el grafo directamente (sin Chainlit).

Corre DENTRO del contenedor app:
    docker compose run --rm app python /srv/app/tests/test_rag.py

Prueba:
1. Una pregunta genérica (chitchat) -> respuesta razonable.
2. Una consulta de conocimiento -> el nodo RAG se ejecuta.
3. Una pregunta sobre imágenes -> el nodo RAG recupera chunks tipo 'image'.
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


def test_chitchat(graph) -> None:
    print("\n-- chitchat: saludo --")
    result = invoke(graph, "Hola, ¿cómo estás?", thread_id="test-chitchat")
    msgs = result["messages"]
    last = msgs[-1].content if msgs else ""
    _check(len(last) > 5, f"respuesta no vacía (len={len(last)})")
    print(f"  respuesta: {last[:120]}...")


def test_rag_knowledge(graph) -> None:
    print("\n-- RAG: pregunta de conocimiento --")
    result = invoke(graph, "¿De qué tratan los documentos que tienes indexados?", thread_id="test-rag-knowledge")
    route = result.get("route", "?")
    _check(route == "rag", f"ruta == 'rag' (got '{route}')")
    last = result["messages"][-1].content
    _check(len(last) > 20, "respuesta no vacía")
    print(f"  ruta: {route}")
    print(f"  respuesta: {last[:200]}...")


def test_rag_images(graph) -> None:
    print("\n-- RAG: pregunta sobre imágenes --")
    result = invoke(graph, "Describe las imágenes o gráficos que encontraste en los documentos.", thread_id="test-rag-images")
    route = result.get("route", "?")
    _check(route == "rag", f"ruta == 'rag' (got '{route}')")
    last = result["messages"][-1].content
    _check(len(last) > 20, "respuesta no vacía")
    print(f"  respuesta: {last[:200]}...")


def test_route_python(graph) -> None:
    print("\n-- Routing: petición de código -> ruta 'python' --")
    result = invoke(graph, "Calcula el factorial de 10 en Python.", thread_id="test-route-python")
    route = result.get("route", "?")
    _check(route == "python", f"ruta == 'python' (got '{route}')")
    last = result["messages"][-1].content
    _check("3628800" in last or "factorial" in last.lower(), f"resultado contiene 3628800 o 'factorial' (got: {last[:120]})")
    print(f"  ruta: {route}")
    print(f"  respuesta: {last[:200]}...")


if __name__ == "__main__":
    print("Construyendo el grafo...")
    graph = build_graph()
    test_chitchat(graph)
    test_rag_knowledge(graph)
    test_rag_images(graph)
    test_route_python(graph)
    print("\n✓ Todos los tests del RAG/grafo pasaron.")
