from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol, cast, runtime_checkable

_config = import_module("pdf_parser.config")
_chunker = import_module("pdf_parser.chunker")
_qdrant_client = import_module("qdrant_client")
qdrant_models = import_module("qdrant_client.http.models")

Settings = _config.Settings
Chunk = _chunker.Chunk
AsyncQdrantClient = _qdrant_client.AsyncQdrantClient


class VectorStoreError(RuntimeError):
    pass


class VectorStoreConfigurationError(VectorStoreError):
    pass


@runtime_checkable
class QdrantCollectionsAPI(Protocol):
    async def collection_exists(self, collection_name: str, **kwargs: object) -> bool: ...

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Any,
        **kwargs: object,
    ) -> bool: ...

    async def get_collection(self, collection_name: str, **kwargs: object) -> object: ...

    async def delete_collection(self, collection_name: str, **kwargs: object) -> object: ...

    async def upsert(
        self,
        collection_name: str,
        points: list[Any],
        **kwargs: object,
    ) -> object: ...


class _QdrantVectorConfig(Protocol):
    size: int


class _QdrantCollectionParams(Protocol):
    vectors: _QdrantVectorConfig | dict[str, _QdrantVectorConfig] | None


class _QdrantCollectionConfig(Protocol):
    params: _QdrantCollectionParams


class _QdrantCollectionInfo(Protocol):
    config: _QdrantCollectionConfig


@dataclass(frozen=True)
class _VectorStoreConfig:
    url: str
    collection_name: str
    dimension: int


class QdrantVectorStore:
    def __init__(
        self,
        settings: Settings,
        *,
        client: QdrantCollectionsAPI | None = None,
    ) -> None:
        self._config = _VectorStoreConfig(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            dimension=settings.embedding_dim,
        )
        self._client: QdrantCollectionsAPI | AsyncQdrantClient = client or AsyncQdrantClient(
            url=self._config.url,
            api_key=settings.qdrant_api_key,
        )
        self._owns_client = client is None
        self._collection_verified = False

    async def aclose(self) -> None:
        if not self._owns_client:
            return

        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    async def wipe_collection(self, *, confirm_collection: str) -> None:
        if confirm_collection != self._config.collection_name:
            raise VectorStoreError(
                "confirmation does not match configured collection: "
                f"expected {self._config.collection_name!r}, got {confirm_collection!r}"
            )

        if not await self._client.collection_exists(self._config.collection_name):
            return

        await self._client.delete_collection(self._config.collection_name)
        self._collection_verified = False

    async def upsert_chunks(
        self,
        *,
        input_id: str,
        document_source: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
        embedding_model: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "chunk and embedding count mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        await self._ensure_collection()
        points = self._build_points(
            input_id=input_id,
            document_source=document_source,
            chunks=chunks,
            embeddings=embeddings,
            source=source,
            embedding_model=embedding_model,
        )

        if not points:
            return 0

        await self._client.upsert(
            collection_name=self._config.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    async def _ensure_collection(self) -> None:
        if self._collection_verified:
            return

        collection_name = self._config.collection_name
        if not await self._client.collection_exists(collection_name):
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self._config.dimension,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            self._collection_verified = True
            return

        collection_info = await self._client.get_collection(collection_name)
        vector_size = _get_collection_vector_size(collection_info)
        if vector_size != self._config.dimension:
            raise VectorStoreConfigurationError(
                "existing Qdrant collection vector size mismatch: "
                f"expected {self._config.dimension}, got {vector_size}"
            )

        self._collection_verified = True

    def _build_points(
        self,
        *,
        input_id: str,
        document_source: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        source: Literal["text", "audio"],
        embedding_model: str,
    ) -> list[Any]:
        points: list[Any] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            actual_dimension = len(embedding)
            if actual_dimension != self._config.dimension:
                raise VectorStoreError(
                    "embedding dimension mismatch for chunk "
                    f"{chunk.chunk_index}: expected {self._config.dimension}, "
                    f"got {actual_dimension}"
                )

            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{input_id}:{chunk.chunk_index}")),
                    vector=embedding,
                    payload={
                        "input_id": input_id,
                        "document_source": document_source,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "source": source,
                        "embedding_model": embedding_model,
                        "semantic_chunking": {
                            "break_threshold": _get_required_chunk_metadata(
                                chunk.metadata,
                                "similarity_threshold",
                            ),
                            "overlap_sentences": _get_required_chunk_metadata(
                                chunk.metadata,
                                "overlap_sentences",
                            ),
                        },
                    },
                )
            )

        return points


def _get_collection_vector_size(collection_info: object) -> int:
    typed_collection_info = cast(_QdrantCollectionInfo, collection_info)
    try:
        vectors_config = typed_collection_info.config.params.vectors
    except AttributeError as exc:
        raise VectorStoreConfigurationError(
            "existing Qdrant collection vector configuration is unavailable"
        ) from exc

    if vectors_config is None:
        raise VectorStoreConfigurationError("existing Qdrant collection has no vector config")

    if isinstance(vectors_config, dict):
        raise VectorStoreConfigurationError(
            "named Qdrant vectors are not supported for this loader collection"
        )

    try:
        return int(vectors_config.size)
    except AttributeError as exc:
        raise VectorStoreConfigurationError(
            "existing Qdrant collection vector size is unavailable"
        ) from exc


def _get_required_chunk_metadata(metadata: dict[str, object], key: str) -> object:
    if key not in metadata:
        raise VectorStoreError(f"semantic chunk metadata missing required key: {key}")

    return metadata[key]
