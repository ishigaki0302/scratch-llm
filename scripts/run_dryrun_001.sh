#!/usr/bin/env bash
# dryrun-001: train 100 steps, stop, resume to 200 (checks checkpoint/resume), then sample.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/training/dryrun-001-300m-ctx4k.json
TR=".venv/bin/torchrun --nproc_per_node=2"
$TR scripts/pretrain.py --config $CFG --max-steps-override 100
$TR scripts/pretrain.py --config $CFG --resume
CK=checkpoints/dryrun-001-300m-ctx4k/$(cat checkpoints/dryrun-001-300m-ctx4k/latest)
.venv/bin/python scripts/generate_samples.py --checkpoint $CK --output runs/dryrun-001-300m-ctx4k/samples.jsonl
echo DRYRUN_DONE
