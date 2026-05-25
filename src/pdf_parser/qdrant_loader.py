from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol, runtime_checkable

from pdf_parser.chunker import Chunk, ChunkDocument
from pdf_parser.config import Settings, get_settings
from pdf_parser.embeddings import DeterministicEmbeddingClient, EmbeddingClient
from pdf_parser.vector_store import QdrantVectorStore


class QdrantLoaderError(RuntimeError):
    pass


class QdrantLoaderConfigurationError(QdrantLoaderError):
    pass


@dataclass(frozen=True)
class QdrantLoadResult:
    chunks: int


@runtime_checkable
class EmbeddingAPI(Protocol):
    @property
    def model_name(self) -> str: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


class QdrantChunkLoader:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embeddings: EmbeddingAPI | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._owns_embeddings = embeddings is None
        self._owns_vector_store = vector_store is None

    async def aclose(self) -> None:
        if self._embeddings is not None and self._owns_embeddings:
            await self._embeddings.aclose()
        if self._vector_store is not None and self._owns_vector_store:
            await self._vector_store.aclose()

    @property
    def collection_name(self) -> str:
        return self._settings.qdrant_collection

    async def wipe(self, *, confirm_collection: str) -> None:
        vector_store = self._get_vector_store()
        await vector_store.wipe_collection(confirm_collection=confirm_collection)

    async def load_file(self, json_path: Path, *, batch_size: int = 64) -> QdrantLoadResult:
        document = load_chunk_document(json_path)
        return await self.load_document(document, batch_size=batch_size)

    async def load_document(
        self,
        document: ChunkDocument,
        *,
        batch_size: int = 64,
    ) -> QdrantLoadResult:
        if document.chunk_count != len(document.chunks):
            raise QdrantLoaderError(
                f"chunk_count mismatch: expected {document.chunk_count}, got {len(document.chunks)}"
            )

        total_chunks = 0
        embeddings = self._get_embeddings()
        vector_store = self._get_vector_store()
        for chunk_batch in _batched(document.chunks, batch_size):
            batch_embeddings = await embeddings.embed_texts([chunk.text for chunk in chunk_batch])
            total_chunks += await vector_store.upsert_chunks(
                input_id=document.source,
                document_source=document.source,
                chunks=list(chunk_batch),
                embeddings=batch_embeddings,
                source="text",
                embedding_model=embeddings.model_name,
            )

        return QdrantLoadResult(chunks=total_chunks)

    def _get_embeddings(self) -> EmbeddingAPI:
        if self._embeddings is None:
            if self._settings.azure_openai_endpoint and self._settings.azure_openai_api_key:
                self._embeddings = EmbeddingClient(self._settings)
            else:
                self._embeddings = DeterministicEmbeddingClient(
                    dimension=self._settings.embedding_dim,
                    model_name=self._settings.embedding_model,
                )
        return self._embeddings

    def _get_vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(self._settings)
        return self._vector_store


def load_chunk_document(json_path: Path) -> ChunkDocument:
    source_path = Path(json_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Chunk JSON not found: {source_path}")

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QdrantLoaderError("chunk JSON must be an object")

    raw_source = payload.get("source")
    raw_chunk_count = payload.get("chunk_count")
    raw_chunks = payload.get("chunks")

    if not isinstance(raw_source, str) or not raw_source.strip():
        raise QdrantLoaderError("chunk JSON is missing a valid source")

    if not isinstance(raw_chunk_count, int):
        raise QdrantLoaderError("chunk JSON is missing a valid chunk_count")

    if not isinstance(raw_chunks, list):
        raise QdrantLoaderError("chunk JSON is missing a valid chunks array")

    chunks: list[Chunk] = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise QdrantLoaderError("chunk JSON contains a non-object chunk entry")

        text = raw_chunk.get("text")
        chunk_index = raw_chunk.get("chunk_index")
        metadata = raw_chunk.get("metadata")
        if not isinstance(text, str) or not text.strip():
            raise QdrantLoaderError("chunk JSON contains a chunk with invalid text")
        if not isinstance(chunk_index, int):
            raise QdrantLoaderError("chunk JSON contains a chunk with invalid chunk_index")
        if not isinstance(metadata, dict):
            raise QdrantLoaderError("chunk JSON contains a chunk with invalid metadata")

        chunks.append(
            Chunk(
                text=text,
                chunk_index=chunk_index,
                metadata=metadata,
            )
        )

    return ChunkDocument(source=raw_source, chunk_count=raw_chunk_count, chunks=chunks)


def _batched(items: list[Chunk], batch_size: int):
    if batch_size <= 0:
        raise QdrantLoaderError("batch_size must be greater than zero")

    iterator = iter(items)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch
