"""Nodo que genera y ejecuta código Python para resolver la petición.

ATENCIÓN: ejecuta código generado por el LLM. Se ejecuta dentro del
contenedor de la app con una guarda ligera (timeout + bloqueo de imports
de sistema/red, ver ``python_guard.py``), pero no es un sandbox real; no
exponer el servicio a usuarios no confiables sin uno (firejail, gVisor,
contenedor efímero, etc.). Documentado como limitación conocida.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ..llm import get_chat_model
from .python_guard import GuardedPythonREPLTool
from .state import GraphState

python_tool = GuardedPythonREPLTool()
TOOLS = [python_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

PYTHON_SYSTEM = (
    "Eres un asistente que resuelve tareas escribiendo y ejecutando código Python. "
    "Cuando necesites calcular o programar algo, usa la herramienta de Python. "
    "Asegurate de usar siempre print() para mostrar el resultado en el mensaje final. "
    "El código ejecutado y su salida se mostrarán automáticamente al usuario antes "
    "de tu respuesta: no los repitas ni los resumas, limítate a explicar el "
    "resultado en lenguaje natural y en español."
)


def python_node(state: GraphState) -> dict:
    model = get_chat_model().bind_tools(TOOLS)
    messages = [SystemMessage(content=PYTHON_SYSTEM), *state["messages"]]

    response = model.invoke(messages)
    new_messages = [response]
    executions: list[tuple[str, str]] = []

    for _ in range(3):
        if not getattr(response, "tool_calls", None):
            break
        messages.append(response)
        for call in response.tool_calls:
            tool = TOOLS_BY_NAME[call["name"]]
            code = str(call["args"].get("query", ""))
            result = tool.invoke(call["args"])
            executions.append((code, str(result)))
            tool_msg = ToolMessage(content=str(result), tool_call_id=call["id"])
            messages.append(tool_msg)
            new_messages.append(tool_msg)
        response = model.invoke(messages)
        new_messages.append(response)

    # El modelo (2B) a veces termina sin texto (content vacío, con o sin
    # tool_calls pendientes) aunque el resultado ya esté en el ToolMessage
    # anterior. Forzamos una respuesta final sin tools (no puede volver a
    # llamar a la herramienta) para garantizar la explicación que pide
    # PYTHON_SYSTEM.
    if not str(new_messages[-1].content or "").strip():
        new_messages[-1] = get_chat_model().invoke(messages)

    # No confiamos en que el modelo (2B) transcriba fielmente el código y la
    # salida en su respuesta final (tiende a parafrasearlos o truncarlos):
    # los anteponemos de forma literal, ya que los tenemos disponibles.
    if executions:
        blocks = "\n\n".join(
            f"```python\n{code}\n```\n**Salida:**\n```\n{output}\n```"
            for code, output in executions
        )
        explanation = str(new_messages[-1].content or "").strip()
        new_messages[-1] = AIMessage(content=f"{blocks}\n\n{explanation}".strip())

    return {"messages": new_messages, "retrieved_image_paths": []}
