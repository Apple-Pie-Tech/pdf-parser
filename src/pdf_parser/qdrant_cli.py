from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pdf_parser.qdrant_loader import QdrantChunkLoader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdrant-loader",
        description="Load semantic chunk JSON into Qdrant or wipe the configured collection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    load_parser = subparsers.add_parser("load", help="Load chunk JSON into Qdrant")
    load_parser.add_argument("json_path", type=Path, help="Path to the chunk JSON file")
    load_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed and upsert at a time",
    )

    wipe_parser = subparsers.add_parser("wipe", help="Delete the configured collection")
    wipe_parser.add_argument(
        "--confirm-collection",
        required=True,
        help="Exact collection name to confirm the destructive delete",
    )

    return parser


async def _run(args: argparse.Namespace) -> int:
    loader = QdrantChunkLoader()
    try:
        if args.command == "load":
            result = await loader.load_file(args.json_path, batch_size=args.batch_size)
            print(f"loaded {result.chunks} chunks into {loader.collection_name}")
            return 0

        await loader.wipe(confirm_collection=args.confirm_collection)
        print(f"wiped collection {loader.collection_name}")
        return 0
    finally:
        await loader.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
