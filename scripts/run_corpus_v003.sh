#!/usr/bin/env bash
# corpus-v003: FineWeb-2 jpn_Jpan (4 shards) -> JSONL -> cleaned -> tokenized (tokenizer v001).
# Wikipedia stays in corpus-v002; the two are mixed at training time (data.train in the config).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/convert_fineweb2.py \
  --shards 000_00000.parquet 000_00001.parquet 000_00002.parquet 000_00003.parquet
$PY scripts/clean_corpus.py --manifest data/manifests/corpus-v003.jsonl \
  --output data/cleaned/corpus-v003-cleaned.jsonl \
  --report data/reports/corpus-v003-cleaning-report.json --min-chars 200 --min-ja-ratio 0.3
rm data/raw/fineweb2_jpn/fineweb2-jpn.jsonl  # parquet is the source of truth; save ~50GB
$PY scripts/tokenize_corpus.py --input data/cleaned/corpus-v003-cleaned.jsonl \
  --output-dir data/tokenized/corpus-v003-spm48k --workers 20
echo PIPELINE_DONE
