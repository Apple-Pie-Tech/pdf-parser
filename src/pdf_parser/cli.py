from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf_parser.parser import parse_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-parser",
        description="Convert a local PDF into plain text with optional Docling JSON output.",
    )
    parser.add_argument("input_pdf", type=Path, help="Path to the local PDF file")
    parser.add_argument(
        "--text",
        type=Path,
        help="Optional path to write the plain text output",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write the structured Docling JSON output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = parse_pdf(args.input_pdf, text_path=args.text, json_path=args.json)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.text is None:
        print(text)

    return 0
