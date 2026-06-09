"""Acceso a Chroma en modo servidor (una colección con texto + imágenes)."""
from __future__ import annotations

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma

from .config import settings
from .embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
    )
