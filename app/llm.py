"""Acceso al LLM (Gemma 4 vía vLLM, API OpenAI-compatible).

Expone:
- ``get_chat_model`` para el grafo (texto + tool-calling).
- ``describe_image`` / ``ask_about_image`` para la visión multimodal usada en la ingesta.
"""
from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .config import settings


def get_chat_model(temperature: float = 0.2, **kwargs) -> ChatOpenAI:
    """Modelo de chat conectado a vLLM. Soporta ``bind_tools`` y visión."""
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
        temperature=temperature,
        **kwargs,
    )


def image_data_uri(image_path: str | Path) -> str:
    raw = Path(image_path).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    suffix = Path(image_path).suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{b64}"


def ask_about_image(image_path: str | Path, prompt: str, temperature: float = 0.0) -> str:
    """Pregunta libre sobre una imagen usando la visión nativa de Gemma."""
    model = get_chat_model(temperature=temperature)
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
        ]
    )
    return model.invoke([message]).content.strip()


def describe_image(image_path: str | Path) -> str:
    """Descripción densa de una imagen para indexarla en el RAG."""
    prompt = (
        "Describe esta imagen en detalle para poder buscarla después. "
        "Menciona objetos, personas, texto visible, gráficos o diagramas y el contexto. "
        "Responde en español, en un único párrafo."
    )
    return ask_about_image(image_path, prompt)
