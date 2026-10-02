#!/usr/bin/env python3
"""Create a tiny local seed corpus from the exported slide text.

This is only for pipeline sanity checks. It is not pretraining data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="slides-text.txt")
    parser.add_argument("--output", default="data/raw/local/slides-text.jsonl")
    parser.add_argument("--manifest", default="data/manifests/corpus-v001.jsonl")
    return parser.parse_args()


def read_slide_pages(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    pages = [page.strip() for page in text.split("\f")]
    return [page for page in pages if page]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    pages = read_slide_pages(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    char_count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for index, page in enumerate(pages, start=1):
            text = "\n".join(line.rstrip() for line in page.splitlines()).strip()
            char_count += len(text)
            record = {
                "id": f"local_slides_text_{index:03d}",
                "source_id": "local_slides_text",
                "text": text,
                "meta": {
                    "slide_index": index,
                    "created_at": created_at,
                    "purpose": "pipeline_sanity_check",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_record = {
        "source_id": "local_slides_text",
        "name": "Scratch LLM deck extracted text",
        "path": str(output_path),
        "format": "jsonl",
        "language": "ja",
        "license": "internal_working_material",
        "include": True,
        "document_count": len(pages),
        "character_count": char_count,
        "created_at": created_at,
        "notes": "Seed corpus for pipeline sanity checks only. Not intended for real pretraining.",
    }
    manifest_path.write_text(
        json.dumps(manifest_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(pages)} documents to {output_path}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

