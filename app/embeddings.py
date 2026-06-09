"""Embeddings multilingües (BGE-m3), con GPU si está disponible.

vLLM sirve un modelo generativo, no produce embeddings; por eso el RAG calcula
sus propios vectores aquí, dentro del contenedor de la app.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando embeddings en {device.upper()}...")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
