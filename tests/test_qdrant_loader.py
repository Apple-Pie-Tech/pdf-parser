from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import cast

import pytest


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_load_module("pdf_parser.chunker", "src/pdf_parser/chunker.py")
_load_module("pdf_parser.config", "src/pdf_parser/config.py")
_load_module("pdf_parser.embeddings", "src/pdf_parser/embeddings.py")
_load_module("pdf_parser.vector_store", "src/pdf_parser/vector_store.py")
loader = _load_module("pdf_parser.qdrant_loader", "src/pdf_parser/qdrant_loader.py")


def test_load_chunk_document_parses_json(tmp_path: Path):
    payload = {
        "source": "feynman.txt",
        "chunk_count": 2,
        "chunks": [
            {
                "text": "First chunk",
                "chunk_index": 0,
                "metadata": {"similarity_threshold": 0.8, "overlap_sentences": 1},
            },
            {
                "text": "Second chunk",
                "chunk_index": 1,
                "metadata": {"similarity_threshold": 0.8, "overlap_sentences": 1},
            },
        ],
    }
    json_path = tmp_path / "chunks.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    document = loader.load_chunk_document(json_path)

    assert document.source == "feynman.txt"
    assert document.chunk_count == 2
    assert [chunk.text for chunk in document.chunks] == ["First chunk", "Second chunk"]


def test_load_chunk_document_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        loader.load_chunk_document(tmp_path / "missing.json")


@pytest.mark.asyncio
async def test_loader_batches_embeddings_and_upserts():
    document = loader.ChunkDocument(
        source="feynman.txt",
        chunk_count=3,
        chunks=[
            loader.Chunk(text="one", chunk_index=0, metadata={"similarity_threshold": 0.8, "overlap_sentences": 1}),
            loader.Chunk(text="two", chunk_index=1, metadata={"similarity_threshold": 0.8, "overlap_sentences": 1}),
            loader.Chunk(text="three", chunk_index=2, metadata={"similarity_threshold": 0.8, "overlap_sentences": 1}),
        ],
    )

    class FakeEmbeddings:
        model_name = "embedding-deployment"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return [[float(index + 1)] * 4 for index, _ in enumerate(texts)]

        async def aclose(self) -> None:
            return None

    class FakeVectorStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def upsert_chunks(self, **kwargs: object) -> int:
            self.calls.append(kwargs)
            return len(cast(list[object], kwargs["chunks"]))

        async def aclose(self) -> None:
            return None

        async def wipe_collection(self, *, confirm_collection: str) -> None:
            raise AssertionError("unexpected wipe")

    fake_embeddings = FakeEmbeddings()
    fake_vector_store = FakeVectorStore()
    qdrant_loader = loader.QdrantChunkLoader(
        settings=loader.Settings(
            qdrant_url="http://qdrant:6333",
            qdrant_api_key="secret",
            qdrant_collection="apple_pie_story_chunks",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret-key",
            azure_openai_api_version="2024-10-21",
            azure_openai_embeddings_deployment="embeddings-deployment",
            embedding_model="text-embedding-3-large",
            embedding_dim=4,
        ),
        embeddings=fake_embeddings,
        vector_store=fake_vector_store,
    )

    result = await qdrant_loader.load_document(document, batch_size=2)

    assert result.chunks == 3
    assert fake_embeddings.calls == [["one", "two"], ["three"]]
    assert [len(cast(list[object], call["chunks"])) for call in fake_vector_store.calls] == [2, 1]


@pytest.mark.asyncio
async def test_loader_rejects_chunk_count_mismatch():
    document = loader.ChunkDocument(
        source="feynman.txt",
        chunk_count=2,
        chunks=[
            loader.Chunk(text="one", chunk_index=0, metadata={"similarity_threshold": 0.8, "overlap_sentences": 1}),
        ],
    )

    class FakeEmbeddings:
        model_name = "embedding-deployment"

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("unexpected embed")

        async def aclose(self) -> None:
            return None

    class FakeVectorStore:
        async def upsert_chunks(self, **kwargs: object) -> int:
            raise AssertionError("unexpected upsert")

        async def aclose(self) -> None:
            return None

        async def wipe_collection(self, *, confirm_collection: str) -> None:
            return None

    qdrant_loader = loader.QdrantChunkLoader(
        settings=loader.Settings(
            qdrant_url="http://qdrant:6333",
            qdrant_api_key="secret",
            qdrant_collection="apple_pie_story_chunks",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret-key",
            azure_openai_api_version="2024-10-21",
            azure_openai_embeddings_deployment="embeddings-deployment",
            embedding_model="text-embedding-3-large",
            embedding_dim=4,
        ),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(loader.QdrantLoaderError):
        await qdrant_loader.load_document(document)
