#!/usr/bin/env python3
"""Summarize a JSONL corpus file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    docs = 0
    chars = 0
    max_chars = 0
    min_chars = None

    with input_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            length = len(record.get("text", ""))
            docs += 1
            chars += length
            max_chars = max(max_chars, length)
            min_chars = length if min_chars is None else min(min_chars, length)

    summary = {
        "path": str(input_path),
        "documents": docs,
        "characters": chars,
        "min_document_chars": min_chars or 0,
        "max_document_chars": max_chars,
        "average_document_chars": round(chars / docs, 2) if docs else 0,
        "rough_token_estimate_ja": round(chars / 1.7) if chars else 0,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"wrote summary to {output_path}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

