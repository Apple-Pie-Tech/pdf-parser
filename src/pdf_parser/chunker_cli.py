from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf_parser.chunker import chunk_text_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="txt-chunker",
        description="Convert a local text file into semantic chunks JSON.",
    )
    parser.add_argument("input_text", type=Path, help="Path to the local text file")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to write the semantic chunks JSON output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        chunk_text_file(args.input_text, json_path=args.json)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0
