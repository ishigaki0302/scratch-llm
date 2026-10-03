#!/usr/bin/env bash
# torch.compile throughput bench in the separate .venv-compile env (torch 2.7.1+cu126).
# Usage: scripts/bench_compile.sh [eager layers model]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv-compile/bin}
for mode in "${@:-eager layers model}"; do
  for m in $mode; do
    echo "=== $m $(date)"
    # Keep the full output per mode: tracebacks from torchrun children come before the summary.
    $PY/torchrun --nproc_per_node=2 scripts/pretrain.py \
      --config configs/training/bench/bench-compile-torch27-$m.json > logs/bench-compile-torch27-$m.log 2>&1
    echo "exit=$? (full log: logs/bench-compile-torch27-$m.log)"
    rm -rf checkpoints/bench/bench-compile-torch27-$m
  done
done
echo BENCH_DONE
