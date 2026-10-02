#!/usr/bin/env python3
"""Validate CPU-only preparation artifacts before any GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "pyproject.toml",
    "uv.lock",
    "configs/corpus-sources.jsonl",
    "configs/tokenizer/tokenizer-v001.json",
    "configs/training/run-001-300m-ctx4k.json",
    "configs/eval/prompt-set-v001.jsonl",
    "data/manifests/corpus-v001.jsonl",
    "data/raw/local/slides-text.jsonl",
    "data/cleaned/corpus-v001-cleaned.jsonl",
    "data/interim/tokenizer-corpus.txt",
    "data/reports/corpus-v001-cleaning-report.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
    return records


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}")
        raise SystemExit(1)

    manifest = read_jsonl(root / "data/manifests/corpus-v001.jsonl")
    cleaned = read_jsonl(root / "data/cleaned/corpus-v001-cleaned.jsonl")
    prompts = read_jsonl(root / "configs/eval/prompt-set-v001.jsonl")

    if not manifest:
        raise SystemExit("manifest is empty")
    if not cleaned:
        raise SystemExit("cleaned corpus is empty")
    if not prompts:
        raise SystemExit("eval prompt set is empty")

    tokenizer_corpus = root / "data/interim/tokenizer-corpus.txt"
    tokenizer_chars = len(tokenizer_corpus.read_text(encoding="utf-8"))
    if tokenizer_chars == 0:
        raise SystemExit("tokenizer corpus is empty")

    print("CPU prep validation passed")
    print(f"manifest_records={len(manifest)}")
    print(f"cleaned_documents={len(cleaned)}")
    print(f"eval_prompts={len(prompts)}")
    print(f"tokenizer_corpus_chars={tokenizer_chars}")


if __name__ == "__main__":
    main()

