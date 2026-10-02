# Pipeline and Data Prep

## What Was Set Up

The project now has a minimal `uv`-based preparation pipeline for corpus handling.

The first pass is intentionally small. It uses the extracted slide text as a seed corpus so the data path can be tested before downloading or processing large external datasets.

## Directory Layout

- `configs/` - source, tokenizer, training, and eval configuration
- `data/raw/` - raw converted source documents
- `data/manifests/` - corpus manifests
- `data/cleaned/` - cleaned JSONL documents
- `data/interim/` - temporary preparation artifacts such as tokenizer input text
- `data/tokenized/` - future tokenized shards
- `data/reports/` - cleaning and corpus summaries
- `scripts/` - data preparation scripts

## Environment

Environment management uses `uv`.

Because the default home cache may be unavailable in the sandbox, use:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/prepare_seed_corpus.py
```

`uv.lock` has been generated.

## Current Scripts

### `scripts/prepare_seed_corpus.py`

Reads `slides-text.txt`, splits the extracted deck text by PDF page breaks, and writes JSONL documents.

Outputs:

- `data/raw/local/slides-text.jsonl`
- `data/manifests/corpus-v001.jsonl`

### `scripts/clean_corpus.py`

Reads corpus files listed in the manifest and applies initial cleaning:

- control character removal
- whitespace normalization
- minimum character threshold
- Japanese character ratio threshold
- exact text deduplication

Outputs:

- `data/cleaned/corpus-v001-cleaned.jsonl`
- `data/reports/corpus-v001-cleaning-report.json`

### `scripts/build_tokenizer_corpus.py`

Converts cleaned JSONL documents into plain text for tokenizer training.

Output:

- `data/interim/tokenizer-corpus.txt`

### `scripts/summarize_corpus.py`

Summarizes JSONL corpus files.

## Current Seed Run Result

Source:

- `slides-text.txt`

Raw seed corpus:

- documents: 25
- characters: 28,874
- rough Japanese token estimate: 16,985

Cleaning result:

- sources included: 1
- documents seen: 25
- documents written: 21
- dropped too short: 3
- dropped low Japanese ratio: 1
- characters written: 9,776

Tokenizer corpus:

- documents: 21
- output: `data/interim/tokenizer-corpus.txt`

## Configs Added

- `configs/corpus-sources.jsonl`
- `configs/tokenizer/tokenizer-v001.json`
- `configs/training/run-001-300m-ctx4k.json`
- `configs/eval/prompt-set-v001.jsonl`

## GitHub Preparation

`.gitignore` excludes generated data, checkpoints, local virtual environments, `uv` cache, and exported slide artifacts.

Track these:

- source code
- configs
- docs
- lockfile

Do not track these:

- raw/cleaned/tokenized corpus files
- checkpoints
- model weights
- local `.venv`
- local `.uv-cache`

## Next Steps

1. Add real corpus source converters.
2. Decide the first external datasets and their license policy.
3. Add corpus source manifests with checksums and document counts.
4. Add tokenizer training script once `sentencepiece` is installed.
5. Add dry-run pretraining script after choosing the training stack.

