#!/usr/bin/env bash
# Wait for corpus-v003 (scripts/run_corpus_v003.sh) to finish, then launch run-003
# (with --resume so restarts continue), then generate samples from the final checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=configs/training/run-003-300m-ctx4k-wiki-web.json
RUN=run-003-300m-ctx4k-wiki-web

until grep -q PIPELINE_DONE logs/corpus-v003.log || ! pgrep -f run_corpus_v003.sh > /dev/null; do
  sleep 60
done
grep -q PIPELINE_DONE logs/corpus-v003.log || { echo "corpus-v003 failed; not launching $RUN"; exit 1; }
test -s data/tokenized/corpus-v003-spm48k/train.bin || { echo "missing corpus-v003 train.bin"; exit 1; }
cat data/tokenized/corpus-v003-spm48k/meta.json

echo "launching $RUN $(date)"
.venv/bin/torchrun --nproc_per_node=2 scripts/pretrain.py --config $CONFIG --resume

LATEST=checkpoints/$RUN/$(cat checkpoints/$RUN/latest)
.venv/bin/python scripts/generate_samples.py --checkpoint "$LATEST" \
  --output runs/$RUN/samples-$(basename "$LATEST").jsonl
echo RUN003_DONE
