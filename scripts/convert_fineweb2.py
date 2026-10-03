#!/usr/bin/env python3
"""Convert Hugging Face HuggingFaceFW/fineweb-2 parquet shards into corpus JSONL.

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
    parser.add_argument("--input-dir", default="data/raw/hf/fineweb2_jpn/data/jpn_Jpan/train")
    parser.add_argument("--shards", nargs="*", default=[],
                        help="shard file names to convert (default: all in --input-dir)")
    parser.add_argument("--output", default="data/raw/fineweb2_jpn/fineweb2-jpn.jsonl")
    parser.add_argument("--manifest", default="data/manifests/corpus-v003.jsonl")
    parser.add_argument("--source-id", default="fineweb2_jpn_Jpan")
    return parser.parse_args()


def upsert_manifest(path: Path, entry: dict) -> None:
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records = [r for r in records if r["source_id"] != entry["source_id"]] + [entry]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    shards = [input_dir / s for s in args.shards] if args.shards else sorted(input_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards in {input_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["id", "url", "dump", "date", "language_score", "minhash_cluster_size", "text"]
    docs = 0
    chars = 0
    with output_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=10000, columns=columns):
                for row in batch.to_pylist():
                    text = row["text"]
                    out.write(
                        json.dumps(
                            {
                                "id": f"{args.source_id}_{row['id']}",
                                "source_id": args.source_id,
                                "text": text,
                                "meta": {
                                    "url": row["url"],
                                    "dump": row["dump"],
                                    "date": row["date"],
                                    "language_score": row["language_score"],
                                    "minhash_cluster_size": row["minhash_cluster_size"],
                                },
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
            "name": "FineWeb-2 Japanese (HuggingFaceFW/fineweb-2 jpn_Jpan)",
            "path": str(output_path),
            "format": "jsonl",
            "language": "ja",
            "license": "ODC-By 1.0 (subject to Common Crawl Terms of Use)",
            "include": True,
            "document_count": docs,
            "character_count": chars,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": f"First web source. Shards: {', '.join(s.name for s in shards)}.",
        },
    )
    print(f"wrote {docs:,} documents ({chars:,} chars) to {output_path}")


if __name__ == "__main__":
    main()
