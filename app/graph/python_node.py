"""Nodo que genera y ejecuta código Python para resolver la petición.

ATENCIÓN: ejecuta código generado por el LLM. Se ejecuta aislado dentro del
contenedor de la app; no exponer el servicio a usuarios no confiables sin un
sandbox real (firejail, gVisor, contenedor efímero, etc.). Documentado como
limitación conocida.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_experimental.tools import PythonREPLTool

from ..llm import get_chat_model
from .state import GraphState

python_tool = PythonREPLTool()
TOOLS = [python_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

PYTHON_SYSTEM = (
    "Eres un asistente que resuelve tareas escribiendo y ejecutando código Python. "
    "Cuando necesites calcular algo, usa la herramienta de Python. "
    "Asegurate de usar siempre print() para mostrar el resultado en el mensaje final. "
    "Siempre debes mostrar el resultado de la ejecución del código al usuario, incluso si es un error. "
    "Tras ejecutar, explica el resultado al usuario en lenguaje natural y en español."
)


def python_node(state: GraphState) -> dict:
    model = get_chat_model().bind_tools(TOOLS)
    messages = [SystemMessage(content=PYTHON_SYSTEM), *state["messages"]]

    response = model.invoke(messages)
    new_messages = [response]

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
        response = model.invoke(messages)
        new_messages.append(response)

    return {"messages": new_messages, "retrieved_image_paths": []}
