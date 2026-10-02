#!/usr/bin/env bash
# corpus-v002: Japanese Wikipedia -> JSONL -> cleaned -> tokenizer input -> tokenizer
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/convert_wikipedia.py
$PY scripts/clean_corpus.py --manifest data/manifests/corpus-v002.jsonl \
  --output data/cleaned/corpus-v002-cleaned.jsonl \
  --report data/reports/corpus-v002-cleaning-report.json --min-chars 200 --min-ja-ratio 0.3
$PY scripts/summarize_corpus.py data/cleaned/corpus-v002-cleaned.jsonl || true
$PY scripts/build_tokenizer_corpus.py --input <(shuf -n 400000 --random-source=<(yes) data/cleaned/corpus-v002-cleaned.jsonl) \
  --output data/interim/tokenizer-corpus.txt
$PY scripts/train_tokenizer.py --threads 20
echo PIPELINE_DONE
