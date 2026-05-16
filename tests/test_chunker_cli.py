from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_load_module("pdf_parser.chunker", "src/pdf_parser/chunker.py")
chunker = sys.modules["pdf_parser.chunker"]
cli = _load_module("pdf_parser.chunker_cli", "src/pdf_parser/chunker_cli.py")


def test_cli_help_shows_usage(capsys):
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "txt-chunker" in captured.out


def test_cli_writes_json(monkeypatch, tmp_path: Path):
    input_text = tmp_path / "sample.txt"
    input_text.write_text("content", encoding="utf-8")
    json_path = tmp_path / "chunks.json"

    recorded: dict[str, Path] = {}

    def fake_chunk_text_file(input_path: Path, *, json_path: Path):
        recorded["input_path"] = input_path
        recorded["json_path"] = json_path

    monkeypatch.setattr(cli, "chunk_text_file", fake_chunk_text_file)

    exit_code = cli.main([str(input_text), "--json", str(json_path)])

    assert exit_code == 0
    assert recorded == {"input_path": input_text, "json_path": json_path}


def test_cli_returns_error_code_on_failure(monkeypatch, capsys, tmp_path: Path):
    input_text = tmp_path / "sample.txt"
    input_text.write_text("content", encoding="utf-8")

    def fake_chunk_text_file(input_path: Path, *, json_path: Path):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "chunk_text_file", fake_chunk_text_file)

    exit_code = cli.main([str(input_text), "--json", str(tmp_path / "chunks.json")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: boom" in captured.err


def test_cli_writes_real_chunk_json(monkeypatch, tmp_path: Path):
    input_text = tmp_path / "sample.txt"
    input_text.write_text("First paragraph\n\nSecond paragraph", encoding="utf-8")
    json_path = tmp_path / "chunks.json"

    class FakeAdapter:
        def __init__(self, settings, *, config):
            self.settings = settings
            self.config = config

        def chunk(self, text: str):
            return [
                chunker.Chunk(
                    text="First paragraph Second paragraph",
                    chunk_index=0,
                    metadata={"provider": "chonkie", "chunk_length": 32},
                )
            ]

    monkeypatch.setattr(
        chunker.ChunkerSettings,
        "from_env",
        classmethod(
            lambda cls: chunker.ChunkerSettings(
                azure_openai_endpoint=None,
                azure_openai_api_key=None,
                azure_openai_api_version="2024-02-01",
                azure_openai_embeddings_deployment=None,
                embedding_model="text-embedding-3-large",
                embedding_dim=None,
            )
        ),
    )
    monkeypatch.setattr(
        chunker._SemanticChunkingConfig,
        "from_env",
        classmethod(
            lambda cls: chunker._SemanticChunkingConfig(
                similarity_threshold=0.8,
                similarity_window=1,
                min_chunk_chars=350,
                max_chunk_chars=1400,
            )
        ),
    )
    monkeypatch.setattr(chunker, "SemanticChunkerAdapter", FakeAdapter)

    exit_code = cli.main([str(input_text), "--json", str(json_path)])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["chunk_count"] == 1
    assert payload["chunks"][0]["text"] == "First paragraph Second paragraph"
