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
```
