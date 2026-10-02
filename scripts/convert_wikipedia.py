#!/usr/bin/env python3
"""Convert Hugging Face wikimedia/wikipedia parquet shards into corpus JSONL.

Also appends/replaces the source entry in a corpus manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/hf/wikipedia/20231101.ja")
    parser.add_argument("--output", default="data/raw/wikipedia_ja/wikipedia-ja-20231101.jsonl")
    parser.add_argument("--manifest", default="data/manifests/corpus-v002.jsonl")
    parser.add_argument("--source-id", default="wikipedia_ja_20231101")
    return parser.parse_args()


def upsert_manifest(path: Path, entry: dict) -> None:
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records = [r for r in records if r["source_id"] != entry["source_id"]] + [entry]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    shards = sorted(Path(args.input_dir).glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards in {args.input_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs = 0
    chars = 0
    with output_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            table = pq.read_table(shard, columns=["id", "url", "title", "text"])
            for row in table.to_pylist():
                text = row["text"]
                out.write(
                    json.dumps(
                        {
                            "id": f"{args.source_id}_{row['id']}",
                            "source_id": args.source_id,
                            "text": text,
                            "meta": {"title": row["title"], "url": row["url"]},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                docs += 1
                chars += len(text)
            print(f"{shard.name}: total docs={docs:,} chars={chars:,}", flush=True)

    upsert_manifest(
        Path(args.manifest),
        {
            "source_id": args.source_id,
            "name": "Japanese Wikipedia (wikimedia/wikipedia 20231101.ja)",
            "path": str(output_path),
            "format": "jsonl",
            "language": "ja",
            "license": "CC BY-SA 4.0 / GFDL",
            "include": True,
            "document_count": docs,
            "character_count": chars,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": "First real pretraining source. HF dataset wikimedia/wikipedia, config 20231101.ja.",
        },
    )
    print(f"wrote {docs:,} documents ({chars:,} chars) to {output_path}")


if __name__ == "__main__":
    main()
