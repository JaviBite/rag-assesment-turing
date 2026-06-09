"""Configuración central de la app (leída de variables de entorno)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # vLLM (API OpenAI-compatible)
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "gemma"
    vllm_api_key: str = "EMPTY"  # vLLM no valida la clave, pero el cliente la exige

    # Chroma (modo servidor)
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    collection_name: str = "knowledge"

    # Servicio de detección
    detector_url: str = "http://detector:8000"

    # Embeddings (CPU, multilingüe)
    embedding_model: str = "BAAI/bge-m3"

    # Memoria conversacional
    summary_token_threshold: int = 3000
    keep_last_messages: int = 4  # mensajes recientes que NO se resumen

    # Rutas
    docs_dir: Path = Path("/docs")
    data_dir: Path = Path("/data")

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def extractions_dir(self) -> Path:
        return self.data_dir / "extractions"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"


settings = Settings()
