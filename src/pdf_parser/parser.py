from __future__ import annotations

import importlib
import json
from pathlib import Path


def _build_converter():
    module = importlib.import_module("docling.document_converter")
    return module.DocumentConverter()


def parse_pdf(
    input_path: Path,
    *,
    text_path: Path | None = None,
    json_path: Path | None = None,
) -> str:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF file not found: {source}")

    converter = _build_converter()
    result = converter.convert(source)
    document = result.document

    text = document.export_to_text()

    if text_path is not None:
        destination = Path(text_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")

    if json_path is not None:
        destination = Path(json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = document.export_to_dict()
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return text
