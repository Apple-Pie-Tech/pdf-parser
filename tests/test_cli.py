from __future__ import annotations

import importlib.util
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


_load_module("pdf_parser.parser", "src/pdf_parser/parser.py")
cli = _load_module("pdf_parser.cli", "src/pdf_parser/cli.py")


def test_cli_help_shows_usage(capsys):
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "pdf-parser" in captured.out


def test_cli_writes_stdout_when_no_text_file(monkeypatch, capsys, tmp_path: Path):
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")

    def fake_parse_pdf(input_path: Path, *, text_path: Path | None, json_path: Path | None) -> str:
        assert input_path == input_pdf
        assert text_path is None
        assert json_path is None
        return "Plain text output"

    monkeypatch.setattr(cli, "parse_pdf", fake_parse_pdf)

    exit_code = cli.main([str(input_pdf)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "Plain text output"


def test_cli_returns_error_code_on_failure(monkeypatch, capsys, tmp_path: Path):
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")

    def fake_parse_pdf(input_path: Path, *, text_path: Path | None, json_path: Path | None) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "parse_pdf", fake_parse_pdf)

    exit_code = cli.main([str(input_pdf)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: boom" in captured.err
