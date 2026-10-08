from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


config = _load_module("pdf_parser.config", "src/pdf_parser/config.py")
Settings = config.Settings

_ENV_ISOLATED_VARS = (
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "SEMANTIC_SIMILARITY_THRESHOLD",
    "CHUNK_OVERLAP_SENTENCES",
    "MIN_CHUNK_CHARS",
    "MAX_CHUNK_CHARS",
)


def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for var in _ENV_ISOLATED_VARS:
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None)


def test_embedding_defaults_match_live_azure_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _isolated_settings(monkeypatch)

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dim == 1536
    assert settings.azure_openai_embeddings_deployment == "text-embedding-3-small"


def test_azure_openai_api_version_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _isolated_settings(monkeypatch)

    assert settings.azure_openai_api_version == "2024-10-21"


def test_chunker_settings_have_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _isolated_settings(monkeypatch)

    assert settings.semantic_similarity_threshold == 0.8
    assert settings.chunk_overlap_sentences == 1
    assert settings.min_chunk_chars == 350
    assert settings.max_chunk_chars == 1400


def test_chunker_settings_are_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.65")
    monkeypatch.setenv("CHUNK_OVERLAP_SENTENCES", "3")
    monkeypatch.setenv("MIN_CHUNK_CHARS", "200")
    monkeypatch.setenv("MAX_CHUNK_CHARS", "900")

    settings = Settings(_env_file=None)

    assert settings.semantic_similarity_threshold == 0.65
    assert settings.chunk_overlap_sentences == 3
    assert settings.min_chunk_chars == 200
    assert settings.max_chunk_chars == 900
