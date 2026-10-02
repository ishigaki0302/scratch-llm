#!/usr/bin/env python3
"""Tokenize cleaned JSONL into flat uint16 token shards for pretraining.

Each document is encoded as `<bos> tokens <eos>` and concatenated.
A deterministic hash of the document id routes ~val_ratio of documents to the val split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sentencepiece as spm

_sp: spm.SentencePieceProcessor | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned/corpus-v002-cleaned.jsonl")
    parser.add_argument("--tokenizer", default="artifacts/tokenizers/tokenizer-v001-spm48k.model")
    parser.add_argument("--output-dir", default="data/tokenized/corpus-v002-spm48k")
    parser.add_argument("--val-ratio", type=float, default=0.002)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--chunk-lines", type=int, default=2000)
    return parser.parse_args()


def init_worker(tokenizer_path: str) -> None:
    global _sp
    _sp = spm.SentencePieceProcessor(model_file=tokenizer_path)


def is_val(doc_id: str, val_ratio: float) -> bool:
    h = int.from_bytes(hashlib.md5(doc_id.encode()).digest()[:8], "little")
    return (h % 1_000_000) < val_ratio * 1_000_000


def encode_chunk(job: tuple[list[str], float]) -> tuple[np.ndarray, np.ndarray, int, int]:
    lines, val_ratio = job
    assert _sp is not None
    train, val = [], []
    n_val_docs = 0
    for line in lines:
        rec = json.loads(line)
        ids = [_sp.bos_id()] + _sp.encode(rec["text"]) + [_sp.eos_id()]
        if is_val(rec["id"], val_ratio):
            val.extend(ids)
            n_val_docs += 1
        else:
            train.extend(ids)
    return (
        np.asarray(train, dtype=np.uint16),
        np.asarray(val, dtype=np.uint16),
        len(lines) - n_val_docs,
        n_val_docs,
    )


def chunks(path: Path, size: int, val_ratio: float):
    buf = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= size:
                yield buf, val_ratio
                buf = []
    if buf:
        yield buf, val_ratio


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    assert sp.get_piece_size() < 65536

    stats = {"train_tokens": 0, "val_tokens": 0, "train_docs": 0, "val_docs": 0}
    with (out_dir / "train.bin").open("wb") as ftr, (out_dir / "val.bin").open("wb") as fva:
        with Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
            jobs = chunks(Path(args.input), args.chunk_lines, args.val_ratio)
            for i, (tr, va, ntr, nva) in enumerate(pool.imap(encode_chunk, jobs)):
                tr.tofile(ftr)
                va.tofile(fva)
                stats["train_tokens"] += len(tr)
                stats["val_tokens"] += len(va)
                stats["train_docs"] += ntr
                stats["val_docs"] += nva
                if i % 50 == 0:
                    print(json.dumps(stats), flush=True)

    meta = {
        **stats,
        "dtype": "uint16",
        "input": args.input,
        "tokenizer": args.tokenizer,
        "vocab_size": sp.get_piece_size(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "val_ratio": args.val_ratio,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
