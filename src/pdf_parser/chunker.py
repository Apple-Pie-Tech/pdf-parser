from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path


class SemanticChunkingError(RuntimeError):
    pass


class SemanticChunkingConfigurationError(SemanticChunkingError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class ChunkDocument:
    source: str
    chunk_count: int
    chunks: list[Chunk]


@dataclass(frozen=True)
class ChunkerSettings:
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str
    azure_openai_embeddings_deployment: str | None
    embedding_model: str
    embedding_dim: int | None

    @classmethod
    def from_env(cls) -> ChunkerSettings:
        raw_dimension = os.getenv("EMBEDDING_DIM")
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_openai_embeddings_deployment=os.getenv(
                "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            embedding_dim=int(raw_dimension) if raw_dimension else None,
        )


@dataclass(frozen=True)
class _SemanticChunkingConfig:
    similarity_threshold: float
    similarity_window: int
    min_chunk_chars: int
    max_chunk_chars: int

    @classmethod
    def from_env(cls) -> _SemanticChunkingConfig:
        return cls(
            similarity_threshold=float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.8")),
            similarity_window=int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1")),
            min_chunk_chars=int(os.getenv("MIN_CHUNK_CHARS", "350")),
            max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "1400")),
        )


DEFAULT_SEMANTIC_CHUNKING_CONFIG = _SemanticChunkingConfig(
    similarity_threshold=0.8,
    similarity_window=1,
    min_chunk_chars=350,
    max_chunk_chars=1400,
)


class SemanticChunkerAdapter:
    def __init__(
        self,
        settings: ChunkerSettings,
        *,
        config: _SemanticChunkingConfig | None = None,
        embedding_model: object | None = None,
        threshold: float | None = None,
        similarity_window: int | None = None,
        min_chunk_chars: int | None = None,
        max_chunk_chars: int | None = None,
    ) -> None:
        base_config = config or DEFAULT_SEMANTIC_CHUNKING_CONFIG
        self._config = _SemanticChunkingConfig(
            similarity_threshold=(
                threshold if threshold is not None else base_config.similarity_threshold
            ),
            similarity_window=(
                similarity_window
                if similarity_window is not None
                else base_config.similarity_window
            ),
            min_chunk_chars=(
                min_chunk_chars
                if min_chunk_chars is not None
                else base_config.min_chunk_chars
            ),
            max_chunk_chars=(
                max_chunk_chars
                if max_chunk_chars is not None
                else base_config.max_chunk_chars
            ),
        )

        try:
            chunker_embeddings = embedding_model or _build_azure_embeddings(settings)
            chunker_module = import_module("chonkie")
            semantic_chunker = chunker_module.SemanticChunker
            self._chunker = semantic_chunker(
                embedding_model=chunker_embeddings,
                threshold=self._config.similarity_threshold,
                similarity_window=self._config.similarity_window,
                chunk_size=self._config.max_chunk_chars,
            )
        except Exception as exc:
            raise SemanticChunkingConfigurationError(
                "failed to initialize Chonkie semantic chunker"
            ) from exc

    def chunk(self, text: str) -> list[Chunk]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return []

        chunks = self._chunker.chunk(normalized_text)
        output_chunks = [
            Chunk(
                text=chunk.text,
                chunk_index=index,
                metadata=self._build_metadata(len(chunk.text)),
            )
            for index, chunk in enumerate(chunks)
        ]
        return self._merge_short_chunks(output_chunks)

    def _build_metadata(self, chunk_length: int) -> dict[str, object]:
        return {
            "provider": "chonkie",
            "chunker": "semantic",
            "similarity_threshold": self._config.similarity_threshold,
            "overlap_sentences": self._config.similarity_window,
            "min_chunk_chars": self._config.min_chunk_chars,
            "max_chunk_chars": self._config.max_chunk_chars,
            "chunk_length": chunk_length,
        }

    def _merge_short_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        merged_texts: list[str] = []
        pending_text = ""

        for chunk in chunks:
            pending_text = f"{pending_text} {chunk.text}".strip() if pending_text else chunk.text
            if len(pending_text) >= self._config.min_chunk_chars:
                merged_texts.append(pending_text)
                pending_text = ""

        if pending_text:
            if merged_texts:
                merged_texts[-1] = f"{merged_texts[-1]} {pending_text}".strip()
            else:
                merged_texts.append(pending_text)

        return [
            Chunk(
                text=text,
                chunk_index=index,
                metadata=self._build_metadata(len(text)),
            )
            for index, text in enumerate(merged_texts)
        ]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def chunk_text_file(
    input_path: Path,
    *,
    json_path: Path,
    adapter: SemanticChunkerAdapter | None = None,
) -> ChunkDocument:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Text file not found: {source}")

    text = source.read_text(encoding="utf-8")
    active_adapter = adapter or SemanticChunkerAdapter(
        ChunkerSettings.from_env(),
        config=_SemanticChunkingConfig.from_env(),
    )
    chunks = active_adapter.chunk(text)

    document = ChunkDocument(source=str(source), chunk_count=len(chunks), chunks=chunks)

    destination = Path(json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(document), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return document


def _build_azure_embeddings(settings: ChunkerSettings) -> object:
    if not settings.azure_openai_endpoint:
        raise SemanticChunkingConfigurationError("AZURE_OPENAI_ENDPOINT is required")

    if not settings.azure_openai_api_key:
        raise SemanticChunkingConfigurationError("AZURE_OPENAI_API_KEY is required")

    if not settings.azure_openai_embeddings_deployment:
        raise SemanticChunkingConfigurationError(
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is required"
        )

    try:
        chunker_module = import_module("chonkie")
        azure_embeddings = chunker_module.AzureOpenAIEmbeddings

        return azure_embeddings(
            model=settings.embedding_model,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment=settings.azure_openai_embeddings_deployment,
            dimension=settings.embedding_dim,
        )
    except Exception as exc:
        raise SemanticChunkingConfigurationError(
            "failed to configure Azure OpenAI embeddings for Chonkie"
        ) from exc
