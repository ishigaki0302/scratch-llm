#!/usr/bin/env python3
"""Build a plain-text tokenizer training corpus from cleaned JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned/corpus-v001-cleaned.jsonl")
    parser.add_argument("--output", default="data/interim/tokenizer-corpus.txt")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means all documents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as out:
        for line in src:
            if args.max_docs and written >= args.max_docs:
                break
            record = json.loads(line)
            text = record["text"].strip()
            if not text:
                continue
            out.write(text)
            out.write("\n\n")
            written += 1

    print(f"wrote {written} documents to {output_path}")


if __name__ == "__main__":
    main()

