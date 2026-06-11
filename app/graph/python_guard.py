"""Guarda ligera para la ejecución de código Python generado por el LLM.

No es un sandbox real (sigue ejecutándose en el contenedor de la app, ver
limitación documentada en README.md): añade dos barreras baratas para el mal
uso más común, bloquear imports con acceso a sistema/red y limitar el tiempo
de ejecución.

El timeout se implementa con un hilo daemon (no con
``PythonREPL.run(timeout=...)``, que usa ``multiprocessing`` con ``spawn`` en
Windows e intenta picklear los ``globals`` del módulo, fallando con
``TypeError: cannot pickle 'module' object`` fuera de Docker/Linux). Si se
excede el tiempo, el hilo queda abandonado (daemon) y se devuelve un aviso.
"""
from __future__ import annotations

import ast
import queue
import sys
import threading

from langchain_experimental.tools.python.tool import PythonREPLTool, sanitize_input

EXEC_TIMEOUT_SECONDS = 10

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "ctypes",
    "importlib", "multiprocessing", "threading", "pathlib",
    "requests", "urllib", "http", "ftplib", "smtplib",
}


def _blocked_import(code: str) -> str | None:
    """Devuelve el nombre del primer módulo bloqueado importado, o None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    return root
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in BLOCKED_MODULES:
                return root
    return None


class GuardedPythonREPLTool(PythonREPLTool):
    """``PythonREPLTool`` con timeout y bloqueo de imports peligrosos."""

    def _run(self, query: str, run_manager=None) -> str:
        if self.sanitize_input:
            query = sanitize_input(query)

        blocked = _blocked_import(query)
        if blocked:
            return (
                f"Ejecución bloqueada: importar '{blocked}' no está permitido "
                "por motivos de seguridad."
            )

        # PythonREPL.worker redirige sys.stdout (global) a un StringIO y solo
        # lo restaura si exec() termina. Si abandonamos el hilo por timeout,
        # un bucle infinito lo dejaría secuestrado para siempre: lo
        # restauramos manualmente.
        real_stdout = sys.stdout
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        thread = threading.Thread(
            target=lambda: result_queue.put(self.python_repl.run(query)),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=EXEC_TIMEOUT_SECONDS)
        if thread.is_alive():
            sys.stdout = real_stdout
            return f"Ejecución cancelada: superó el tiempo límite de {EXEC_TIMEOUT_SECONDS}s."
        return result_queue.get()
