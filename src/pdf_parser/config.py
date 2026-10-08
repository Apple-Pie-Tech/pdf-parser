from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        extra="ignore",
    )

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "apple_pie_story_chunks"
    # Attributed to points written by qdrant-loader when --user-id is not given.
    # pdf-parser ingests documents rather than acting on behalf of an end user,
    # so this identifies the ingestion pipeline itself.
    default_user_id: str = "pdf-parser"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embeddings_deployment: str = "text-embedding-3-small"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    semantic_similarity_threshold: float = 0.8
    chunk_overlap_sentences: int = 1
    min_chunk_chars: int = 350
    max_chunk_chars: int = 1400


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
