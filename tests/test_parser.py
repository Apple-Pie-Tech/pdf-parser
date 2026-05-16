from __future__ import annotations

import importlib.util
import json
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


parser = _load_module("pdf_parser.parser", "src/pdf_parser/parser.py")


class FakeDocument:
    def export_to_text(self) -> str:
        return "Heading\n\nBody text"

    def export_to_dict(self) -> dict:
        return {"title": "Heading", "body": ["Body text"]}


class FakeResult:
    document = FakeDocument()


class FakeConverter:
    def convert(self, source: Path) -> FakeResult:
        assert source.name == "sample.pdf"
        return FakeResult()


def test_parse_pdf_writes_text_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")

    text_path = tmp_path / "out" / "result.txt"
    json_path = tmp_path / "out" / "result.json"

    monkeypatch.setattr(parser, "_build_converter", lambda: FakeConverter())

    text = parser.parse_pdf(input_pdf, text_path=text_path, json_path=json_path)

    assert text == "Heading\n\nBody text"
    assert text_path.read_text(encoding="utf-8") == text
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "title": "Heading",
        "body": ["Body text"],
    }


def test_parse_pdf_raises_for_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parser.parse_pdf(tmp_path / "missing.pdf")
