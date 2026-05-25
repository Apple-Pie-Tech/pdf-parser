# pdf-parser

Simple PDF-to-text parser built on top of [Docling](https://github.com/docling-project/docling).

## What it does

- converts a local PDF into well-formatted plain text
- optionally saves the structured Docling document as JSON
- can print text to stdout or write it to a file

## Install

```bash
pip install -e .
```

## Usage

```bash
pdf-parser input.pdf
pdf-parser input.pdf --text output.txt
pdf-parser input.pdf --text output.txt --json output.docling.json

txt-chunker output.txt --json chunks.json
```

## Chunking text into JSON

The `txt-chunker` command reads a plain-text file and writes semantic chunks as JSON.

```bash
txt-chunker output.txt --json chunks.json
```

It uses Chonkie semantic chunking with Azure OpenAI embeddings. Configuration is read from
environment variables:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (optional, default: `2024-02-01`)
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `EMBEDDING_MODEL` (optional, default: `text-embedding-3-large`)
- `EMBEDDING_DIM` (optional)
- `SEMANTIC_SIMILARITY_THRESHOLD` (optional, default: `0.8`)
- `CHUNK_OVERLAP_SENTENCES` (optional, default: `1`)
- `MIN_CHUNK_CHARS` (optional, default: `350`)
- `MAX_CHUNK_CHARS` (optional, default: `1400`)

## Loading chunks into Qdrant

The `qdrant-loader` command can wipe the configured collection and reload chunk JSON.

```bash
qdrant-loader wipe --confirm-collection apple_pie_story_chunks
qdrant-loader load feynman-azure-chunks.json --batch-size 64
```

It reads Qdrant settings from `.env`. If Azure OpenAI settings are not present, the loader falls back to deterministic mock embeddings so the chunk data can still be loaded.
