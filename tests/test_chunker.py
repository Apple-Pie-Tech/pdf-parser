from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


chunker = _load_module("pdf_parser.chunker", "src/pdf_parser/chunker.py")


class FakeAdapter:
    def chunk(self, text: str) -> list[object]:
        assert text == "Section one\n\nSection two"
        return [
            chunker.Chunk(
                text="Section one",
                chunk_index=0,
                metadata={"provider": "chonkie", "chunk_length": 11},
            ),
            chunker.Chunk(
                text="Section two",
                chunk_index=1,
                metadata={"provider": "chonkie", "chunk_length": 11},
            ),
        ]


def test_chunk_text_file_writes_json(tmp_path: Path):
    input_text = tmp_path / "sample.txt"
    input_text.write_text("Section one\n\nSection two", encoding="utf-8")
    json_path = tmp_path / "out" / "chunks.json"

    document = chunker.chunk_text_file(input_text, json_path=json_path, adapter=FakeAdapter())

    assert document.chunk_count == 2
    assert [item.text for item in document.chunks] == ["Section one", "Section two"]

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["source"] == str(input_text)
    assert payload["chunk_count"] == 2
    assert payload["chunks"][0]["chunk_index"] == 0
    assert payload["chunks"][1]["text"] == "Section two"


def test_chunk_text_file_raises_for_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        chunker.chunk_text_file(
            tmp_path / "missing.txt",
            json_path=tmp_path / "chunks.json",
            adapter=FakeAdapter(),
        )


def test_chunker_settings_reads_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "embeddings")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")

    settings = chunker.ChunkerSettings.from_env()

    assert settings.azure_openai_endpoint == "https://example.openai.azure.com"
    assert settings.azure_openai_api_key == "secret"
    assert settings.azure_openai_embeddings_deployment == "embeddings"
    assert settings.azure_openai_api_version == "2024-03-01-preview"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dim == 1536


def test_chunker_settings_from_env_sources_via_config_settings(monkeypatch: pytest.MonkeyPatch):
    """ChunkerSettings.from_env must build on pdf_parser.config.Settings, the same
    configuration source qdrant-loader uses, instead of reading os.environ directly."""

    class FakeSettings:
        def __init__(self) -> None:
            self.azure_openai_endpoint = "https://fake.openai.azure.com"
            self.azure_openai_api_key = "fake-key"
            self.azure_openai_api_version = "2099-01-01"
            self.azure_openai_embeddings_deployment = "fake-deployment"
            self.embedding_model = "fake-model"
            self.embedding_dim = 99

    monkeypatch.setattr(chunker, "Settings", FakeSettings)

    settings = chunker.ChunkerSettings.from_env()

    assert settings == chunker.ChunkerSettings(
        azure_openai_endpoint="https://fake.openai.azure.com",
        azure_openai_api_key="fake-key",
        azure_openai_api_version="2099-01-01",
        azure_openai_embeddings_deployment="fake-deployment",
        embedding_model="fake-model",
        embedding_dim=99,
    )


def test_semantic_chunking_config_from_env_sources_via_config_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    """SEMANTIC_SIMILARITY_THRESHOLD / CHUNK_OVERLAP_SENTENCES / MIN_CHUNK_CHARS /
    MAX_CHUNK_CHARS must be real Settings fields, not bare os.getenv reads."""

    class FakeSettings:
        def __init__(self) -> None:
            self.semantic_similarity_threshold = 0.55
            self.chunk_overlap_sentences = 4
            self.min_chunk_chars = 111
            self.max_chunk_chars = 999

    monkeypatch.setattr(chunker, "Settings", FakeSettings)

    config = chunker._SemanticChunkingConfig.from_env()

    assert config == chunker._SemanticChunkingConfig(
        similarity_threshold=0.55,
        similarity_window=4,
        min_chunk_chars=111,
        max_chunk_chars=999,
    )


def test_semantic_chunker_adapter_merges_short_chunks(monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, object] = {}

    class FakeAzureOpenAIEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            recorded["embedding_kwargs"] = kwargs
            recorded["embedding_instance"] = self

    class FakeSemanticChunker:
        def __init__(self, **kwargs: object) -> None:
            recorded["chunker_kwargs"] = kwargs

        def chunk(self, text: str) -> list[SimpleNamespace]:
            recorded["normalized_text"] = text
            return [
                SimpleNamespace(text="tiny"),
                SimpleNamespace(text="bits"),
                SimpleNamespace(text="this part is long enough"),
            ]

    fake_module = SimpleNamespace(
        AzureOpenAIEmbeddings=FakeAzureOpenAIEmbeddings,
        SemanticChunker=FakeSemanticChunker,
    )

    monkeypatch.setattr(chunker, "import_module", lambda name: fake_module)

    settings = chunker.ChunkerSettings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret",
        azure_openai_api_version="2024-02-01",
        azure_openai_embeddings_deployment="embeddings",
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
    )

    adapter = chunker.SemanticChunkerAdapter(
        settings,
        threshold=0.7,
        similarity_window=2,
        min_chunk_chars=8,
        max_chunk_chars=200,
    )

    chunks = adapter.chunk("tiny    bits\n\nthis part is long enough")

    assert recorded["normalized_text"] == "tiny bits this part is long enough"
    assert recorded["chunker_kwargs"] == {
        "embedding_model": recorded["embedding_instance"],
        "threshold": 0.7,
        "similarity_window": 2,
        "chunk_size": 200,
    }
    assert [item.text for item in chunks] == [
        "tiny bits",
        "this part is long enough",
    ]
    assert chunks[0].metadata["min_chunk_chars"] == 8
