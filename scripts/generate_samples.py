#!/usr/bin/env python3
"""Generate continuation samples from a checkpoint for the eval prompt set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import LlamaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizers/tokenizer-v001-spm48k.model")
    parser.add_argument("--prompts", default="configs/eval/prompt-set-v001.jsonl")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l]

    results = []
    for p in prompts:
        ids = torch.tensor([[sp.bos_id()] + sp.encode(p["prompt"])], device="cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p,
                                 repetition_penalty=1.1, eos_token_id=sp.eos_id(),
                                 pad_token_id=sp.pad_id())
        text = sp.decode(out[0, ids.shape[1]:].tolist())
        results.append({"prompt_id": p["prompt_id"], "prompt": p["prompt"], "completion": text})
        print(f"### {p['prompt_id']}\n{p['prompt']}【→】{text}\n", flush=True)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
