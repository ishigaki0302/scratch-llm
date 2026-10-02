#!/usr/bin/env python3
"""Train a SentencePiece tokenizer from a tokenizer config JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer/tokenizer-v001.json")
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_path = Path(cfg["outputs"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(model_path.with_suffix(""))

    algorithm = cfg["algorithm"].removeprefix("sentencepiece_")
    spm.SentencePieceTrainer.train(
        input=cfg["input"],
        model_prefix=prefix,
        model_type=algorithm,
        vocab_size=cfg["vocab_size"],
        character_coverage=cfg["character_coverage"],
        normalization_rule_name=cfg["normalization"],
        byte_fallback=cfg["byte_fallback"],
        input_sentence_size=cfg.get("input_sentence_size", 5_000_000),
        shuffle_input_sentence=True,
        max_sentence_length=cfg.get("max_sentence_length", 16384),
        split_digits=cfg.get("split_digits", True),
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=cfg.get("user_defined_symbols", []),
        num_threads=args.threads,
        train_extremely_large_corpus=False,
    )
    print(f"wrote {prefix}.model / {prefix}.vocab")


if __name__ == "__main__":
    main()
