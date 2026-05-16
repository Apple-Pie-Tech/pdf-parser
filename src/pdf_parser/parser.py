from __future__ import annotations

import importlib
import json
import re
from pathlib import Path


def _build_converter():
    module = importlib.import_module("docling.document_converter")
    return module.DocumentConverter()


def _normalize_text(text: str) -> str:
    normalized_lines: list[str] = []
    blank_line_count = 0

    for raw_line in text.splitlines():
        collapsed_line = re.sub(r"[\t ]+", " ", raw_line).strip()
        if collapsed_line:
            blank_line_count = 0
            normalized_lines.append(collapsed_line)
            continue

        if blank_line_count == 0:
            normalized_lines.append("")
        blank_line_count += 1

    normalized_text = "\n".join(normalized_lines).strip()
    if text.endswith("\n") and normalized_text:
        return f"{normalized_text}\n"
    return normalized_text


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

    text = _normalize_text(document.export_to_text())

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
