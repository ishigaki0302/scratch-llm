# Scratch LLM

Scratch Japanese SLM preparation project.

Source presentation:
https://docs.google.com/presentation/d/1nJgEltvNbSp5p4-GGDq2VtGXr6rvbMovR3CZI3AA5j0/edit?usp=sharing

## Local Materials

- `slides.pdf` - exported copy of the Google Slides deck
- `slides.pptx` - editable deck export
- `slides-text.txt` - extracted deck text

## Project Layout

- `configs/` - corpus, tokenizer, training, and evaluation configs
- `data/` - generated local data artifacts, ignored by git
- `docs/` - project documentation and dated work logs
- `scripts/` - data preparation utilities
- `src/scratch_llm/` - Python package namespace

## Environment

Use `uv` for environment management.

The sandbox may not allow writing to the default `uv` cache under the home directory, so commands below use a project-local cache:

```bash
UV_CACHE_DIR=.uv-cache uv lock
```

## Current Seed Pipeline

The current pipeline uses `slides-text.txt` only as a tiny seed corpus to verify the data flow.
It is not real pretraining data.

Run the preparation steps:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/prepare_seed_corpus.py
UV_CACHE_DIR=.uv-cache uv run python scripts/clean_corpus.py
UV_CACHE_DIR=.uv-cache uv run python scripts/build_tokenizer_corpus.py
UV_CACHE_DIR=.uv-cache uv run python scripts/summarize_corpus.py data/cleaned/corpus-v001-cleaned.jsonl
```

Generated outputs:

- `data/manifests/corpus-v001.jsonl`
- `data/raw/local/slides-text.jsonl`
- `data/cleaned/corpus-v001-cleaned.jsonl`
- `data/interim/tokenizer-corpus.txt`
- `data/reports/corpus-v001-cleaning-report.json`

## Next Data Work

The next step is to add real corpus sources to `configs/corpus-sources.jsonl` and convert them into JSONL records with this shape:

```json
{"id":"source_doc_id","source_id":"source_name","text":"document text","meta":{}}
```

Then extend `data/manifests/corpus-v001.jsonl` with the generated source files and rerun cleaning.
