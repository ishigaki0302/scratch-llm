#!/usr/bin/env bash
# Wait for dryrun-001 to finish successfully, then launch run-002 (with --resume so restarts continue).
set -euo pipefail
cd "$(dirname "$0")/.."
until grep -qE "DRYRUN_DONE|Traceback|exitcode" logs/dryrun-001.log; do sleep 30; done
grep -q DRYRUN_DONE logs/dryrun-001.log || { echo "dryrun failed; not launching run-002"; exit 1; }
exec .venv/bin/torchrun --nproc_per_node=2 scripts/pretrain.py \
  --config configs/training/run-002-300m-ctx4k-wiki1ep.json --resume
