#!/usr/bin/env python3
"""Clean JSONL corpus records listed in a corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")
JA_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/corpus-v001.jsonl")
    parser.add_argument("--output", default="data/cleaned/corpus-v001-cleaned.jsonl")
    parser.add_argument("--report", default="data/reports/corpus-v001-cleaning-report.json")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--min-ja-ratio", type=float, default=0.1)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc


def normalize_text(text: str) -> str:
    text = CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(MULTI_SPACE.sub(" ", line).strip() for line in text.splitlines())
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def japanese_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    return len(JA_CHARS.findall(text)) / len(visible)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    report_path = Path(args.report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    seen_hashes: set[str] = set()

    with output_path.open("w", encoding="utf-8") as out:
        for source in read_jsonl(manifest_path):
            if not source.get("include", False):
                counters["sources_skipped"] += 1
                continue

            source_path = Path(source["path"])
            counters["sources_included"] += 1

            for record in read_jsonl(source_path):
                counters["documents_seen"] += 1
                text = normalize_text(record.get("text", ""))
                text_hash = digest_text(text)

                if len(text) < args.min_chars:
                    counters["dropped_too_short"] += 1
                    continue

                if japanese_ratio(text) < args.min_ja_ratio:
                    counters["dropped_low_ja_ratio"] += 1
                    continue

                if text_hash in seen_hashes:
                    counters["dropped_duplicate"] += 1
                    continue

                seen_hashes.add(text_hash)
                counters["documents_written"] += 1
                counters["characters_written"] += len(text)

                cleaned = {
                    "id": record["id"],
                    "source_id": record.get("source_id", source["source_id"]),
                    "text": text,
                    "meta": {
                        **record.get("meta", {}),
                        "cleaning": {
                            "text_sha256": text_hash,
                            "japanese_ratio": round(japanese_ratio(text), 4),
                            "char_count": len(text),
                        },
                    },
                }
                out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    report_path.write_text(
        json.dumps(dict(counters), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote cleaned corpus to {output_path}")
    print(f"wrote cleaning report to {report_path}")


if __name__ == "__main__":
    main()

