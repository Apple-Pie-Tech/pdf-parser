from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


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
_load_module("pdf_parser.qdrant_loader", "src/pdf_parser/qdrant_loader.py")
cli = _load_module("pdf_parser.qdrant_cli", "src/pdf_parser/qdrant_cli.py")


def test_cli_help_shows_usage(capsys):
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "qdrant-loader" in captured.out


def test_cli_loads_json(monkeypatch, tmp_path: Path):
    json_path = tmp_path / "chunks.json"
    json_path.write_text(
        '{"source":"feynman.txt","chunk_count":0,"chunks":[]}',
        encoding="utf-8",
    )

    recorded: dict[str, object] = {}

    class FakeLoader:
        def __init__(self) -> None:
            recorded["constructed"] = True
            self.collection_name = "apple_pie_story_chunks"

        async def load_file(self, json_path: Path, *, batch_size: int):
            recorded["load_file"] = (json_path, batch_size)
            return SimpleNamespace(chunks=0)

        async def wipe(self, *, confirm_collection: str):
            raise AssertionError("unexpected wipe")

        async def aclose(self) -> None:
            recorded["closed"] = True

    monkeypatch.setattr(cli, "QdrantChunkLoader", FakeLoader)

    exit_code = cli.main(["load", str(json_path), "--batch-size", "3"])

    assert exit_code == 0
    assert recorded["load_file"] == (json_path, 3)
    assert recorded["closed"] is True


def test_cli_rejects_wipe_without_confirmation(monkeypatch):
    called = {}

    class FakeLoader:
        def __init__(self) -> None:
            pass

        async def load_file(self, json_path: Path, *, batch_size: int):
            raise AssertionError("unexpected load")

        async def wipe(self, *, confirm_collection: str):
            called["wipe"] = confirm_collection

        async def aclose(self) -> None:
            called["closed"] = True

    monkeypatch.setattr(cli, "QdrantChunkLoader", FakeLoader)

    try:
        cli.main(["wipe"])
    except SystemExit as exc:
        assert exc.code == 2

    assert called == {}


def test_cli_wipes_collection(monkeypatch, capsys):
    recorded: dict[str, object] = {}

    class FakeLoader:
        def __init__(self) -> None:
            self.collection_name = "apple_pie_story_chunks"

        async def load_file(self, json_path: Path, *, batch_size: int):
            raise AssertionError("unexpected load")

        async def wipe(self, *, confirm_collection: str):
            recorded["wipe"] = confirm_collection

        async def aclose(self) -> None:
            recorded["closed"] = True

    monkeypatch.setattr(cli, "QdrantChunkLoader", FakeLoader)

    exit_code = cli.main(["wipe", "--confirm-collection", "apple_pie_story_chunks"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wiped collection apple_pie_story_chunks" in captured.out
    assert recorded["wipe"] == "apple_pie_story_chunks"
    assert recorded["closed"] is True
