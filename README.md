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
`.env` (see `.env.example`) and/or environment variables, via the same `pdf_parser.config.Settings`
used by `qdrant-loader`:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (optional, default: `2024-10-21`)
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `EMBEDDING_MODEL` (optional, default: `text-embedding-3-small`)
- `EMBEDDING_DIM` (optional, default: `1536`)
- `SEMANTIC_SIMILARITY_THRESHOLD` (optional, default: `0.8`)
- `CHUNK_OVERLAP_SENTENCES` (optional, default: `1`)
- `MIN_CHUNK_CHARS` (optional, default: `350`)
- `MAX_CHUNK_CHARS` (optional, default: `1400`)
- `DEFAULT_USER_ID` (optional, default: `pdf-parser`)

## Loading chunks into Qdrant

The `qdrant-loader` command can wipe the configured collection and reload chunk JSON.

```bash
qdrant-loader wipe --confirm-collection apple_pie_story_chunks
qdrant-loader load feynman-azure-chunks.json --batch-size 64 --user-id researcher-42
```

It reads Qdrant settings from `.env`. If Azure OpenAI settings are not present, the loader falls back to deterministic mock embeddings so the chunk data can still be loaded.

Points are written into the shared `apple_pie_story_chunks` collection alongside
`data-ingestion`'s. Point IDs are namespaced with a `pdf:` prefix
(`uuid5(NAMESPACE_URL, f"pdf:{input_id}:{chunk_index}")`) so they never collide with
`data-ingestion`'s `uuid5(NAMESPACE_URL, f"{input_id}:{chunk_index}")` scheme. Each
point's payload includes `user_id` (from `--user-id`, falling back to
`DEFAULT_USER_ID`) and `timestamp` (the ingestion time), matching the keys the
downstream `data-provision-api` reader expects.
