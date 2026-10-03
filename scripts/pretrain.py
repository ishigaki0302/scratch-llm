#!/usr/bin/env python3
"""Minimal Llama-style pretraining loop (PyTorch DDP + HF LlamaForCausalLM).

Launch:
  torchrun --nproc_per_node=2 scripts/pretrain.py --config configs/training/<run>.json [--resume]

Data: flat uint16 token files written by scripts/tokenize_corpus.py. Either a single source
  ("train_bin"/"val_bin") or a weighted mixture:
    "train": [{"name": "wiki", "bin": "...", "weight": 0.2}, {"name": "web", ...}]
    "val":   {"wiki": "...", "web": "..."}   (val_loss_<name> is logged for each)
Outputs:
  checkpoints/<run_id>/step_XXXXXXX/  (HF model + train_state.pt)
  checkpoints/<run_id>/latest          (text file with latest step dir name)
  runs/<run_id>/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps-override", type=int, default=0,
                        help="stop after this many steps (for resume tests); schedule unchanged")
    return parser.parse_args()


def setup_dist() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


class TokenSampler:
    """Random seq_len windows; with several sources, each row picks a source by weight."""

    def __init__(self, paths: str | list[str], seq_len: int, seed: int,
                 weights: list[float] | None = None):
        paths = [paths] if isinstance(paths, str) else paths
        self.data = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        w = np.asarray(weights or [1.0] * len(paths), dtype=np.float64)
        self.probs = w / w.sum()
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, bsz: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.data) == 1:
            src = [0] * bsz
        else:
            src = self.rng.choice(len(self.data), size=bsz, p=self.probs)
        rows = []
        for s in src:
            data = self.data[s]
            i = self.rng.integers(0, len(data) - self.seq_len - 1)
            rows.append(data[i : i + self.seq_len + 1].astype(np.int64))
        x = np.stack(rows)
        t = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


def lr_at(step: int, cfg: dict) -> float:
    warmup, total = cfg["warmup_steps"], cfg["max_steps"]
    peak, floor = cfg["learning_rate"], cfg["learning_rate"] * cfg.get("min_lr_ratio", 0.1)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, sampler: TokenSampler, steps: int, bsz: int, device, autocast) -> float:
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = sampler.batch(bsz, device)
        with autocast:
            logits = model(input_ids=x).logits
        losses.append(torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rank, world, local = setup_dist()
    device = torch.device("cuda", local)
    master = rank == 0
    tcfg, mcfg, dcfg = cfg["training"], cfg["model"], cfg["data"]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(tcfg.get("seed", 0))

    run_id = cfg["run_id"]
    ckpt_root = Path(cfg.get("checkpoint_dir", "checkpoints")) / run_id
    run_dir = Path(cfg.get("run_dir", "runs")) / run_id
    if master:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, run_dir / "config.json")

    model_config = LlamaConfig(
        vocab_size=mcfg["vocab_size"],
        hidden_size=mcfg["hidden_size"],
        intermediate_size=mcfg["intermediate_size"],
        num_hidden_layers=mcfg["num_layers"],
        num_attention_heads=mcfg["num_heads"],
        num_key_value_heads=mcfg["num_kv_heads"],
        max_position_embeddings=mcfg["context_length"],
        rope_theta=mcfg.get("rope_theta", 10000.0),
        rms_norm_eps=1e-5,
        tie_word_embeddings=mcfg.get("tie_embeddings", True),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        attn_implementation="sdpa",
    )

    start_step = 0
    resume_dir = None
    if args.resume and (ckpt_root / "latest").exists():
        resume_dir = ckpt_root / (ckpt_root / "latest").read_text().strip()

    if resume_dir is not None:
        model = LlamaForCausalLM.from_pretrained(resume_dir, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa")
    else:
        model = LlamaForCausalLM(model_config)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg["learning_rate"], betas=tuple(tcfg.get("betas", [0.9, 0.95])),
        eps=1e-8, fused=True,
    )

    seq_len = mcfg["context_length"]
    micro_bsz = tcfg["micro_batch_size"]
    grad_accum = tcfg["global_batch_size"] // (micro_bsz * world)
    assert grad_accum * micro_bsz * world == tcfg["global_batch_size"]
    tokens_per_step = tcfg["global_batch_size"] * seq_len

    train_seed = tcfg.get("seed", 0) * 1000 + rank
    if "train" in dcfg:
        train_sampler = TokenSampler([s["bin"] for s in dcfg["train"]], seq_len, seed=train_seed,
                                     weights=[s["weight"] for s in dcfg["train"]])
    else:
        train_sampler = TokenSampler(dcfg["train_bin"], seq_len, seed=train_seed)
    val_bins = dcfg["val"] if "val" in dcfg else {"": dcfg["val_bin"]}
    val_samplers = {name: TokenSampler(path, seq_len, seed=12345 + rank)
                    for name, path in val_bins.items()}

    if resume_dir is not None:
        state = torch.load(resume_dir / "train_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        if len(state["data_rng"]) == world:
            train_sampler.rng.bit_generator.state = state["data_rng"][rank]
        if master:
            print(f"resumed from {resume_dir} at step {start_step}", flush=True)

    raw_model = model
    compile_mode = tcfg.get("compile", False)
    if compile_mode == "model":
        model.compile()
    elif compile_mode:
        # Whole-model compile trips on transformers' output-capturing wrappers; compile per layer.
        for layer in raw_model.model.layers:
            layer.compile()
    if world > 1:
        model = DDP(model, device_ids=[local])

    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    max_steps = tcfg["max_steps"]
    stop_step = min(max_steps, start_step + args.max_steps_override) if args.max_steps_override else max_steps

    if master:
        print(json.dumps({"run_id": run_id, "params": n_params, "world": world,
                          "grad_accum": grad_accum, "tokens_per_step": tokens_per_step,
                          "start_step": start_step, "stop_step": stop_step}), flush=True)
        meta = {"git": git_hash(), "params": n_params, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "torch": torch.__version__, "tokens_per_step": tokens_per_step}
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    metrics_f = open(run_dir / "metrics.jsonl", "a") if master else None

    def save(step: int) -> None:
        rngs = [None] * world
        local_rng = train_sampler.rng.bit_generator.state
        if world > 1:
            dist.all_gather_object(rngs, local_rng)
        else:
            rngs = [local_rng]
        if not master:
            return
        out = ckpt_root / f"step_{step:07d}"
        raw_model.save_pretrained(out, safe_serialization=True)
        torch.save({"optimizer": optimizer.state_dict(), "step": step,
                    "data_rng": rngs, "config": cfg}, out / "train_state.pt")
        (ckpt_root / "latest").write_text(out.name)
        keep = tcfg.get("keep_checkpoints", 3)
        olds = sorted(ckpt_root.glob("step_*"))[:-keep]
        for d in olds:
            shutil.rmtree(d)
        print(f"saved {out}", flush=True)

    model.train()
    t0 = time.time()
    tokens_since = 0
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step, stop_step):
        lr = lr_at(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        for micro in range(grad_accum):
            x, y = train_sampler.batch(micro_bsz, device)
            sync = (micro == grad_accum - 1) or world == 1
            ctx = nullcontext() if sync else model.no_sync()
            with ctx, autocast:
                logits = model(input_ids=x).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), y.reshape(-1)) / grad_accum
            loss.backward()
            loss_acc += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_since += tokens_per_step

        if world > 1:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
        done = step + 1
        if done % tcfg.get("log_interval", 10) == 0 and master:
            torch.cuda.synchronize()
            dt = time.time() - t0
            rec = {"step": done, "loss": round(loss_acc.item(), 4), "lr": lr,
                   "grad_norm": round(grad_norm.item(), 3),
                   "tok_per_s": round(tokens_since / dt),
                   "tokens_seen": done * tokens_per_step,
                   "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                   "time": time.time()}
            print(json.dumps(rec), flush=True)
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            t0, tokens_since = time.time(), 0

        if done % tcfg["eval_interval"] == 0 or done == stop_step:
            vls = {}
            for name, sampler in val_samplers.items():
                vl = evaluate(raw_model, sampler, tcfg.get("eval_steps", 20), micro_bsz, device, autocast)
                if world > 1:
                    t = torch.tensor(vl, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.AVG)
                    vl = t.item()
                vls[name] = vl
            if master:
                vl = float(np.mean(list(vls.values())))
                rec = {"step": done, "val_loss": round(vl, 4), "val_ppl": round(math.exp(vl), 2)}
                if len(vls) > 1 or "" not in vls:
                    rec.update({f"val_loss_{n}": round(v, 4) for n, v in vls.items()})
                rec["time"] = time.time()
                print(json.dumps(rec), flush=True)
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
            t0, tokens_since = time.time(), 0

        if done % tcfg["checkpoint_interval"] == 0 or done == stop_step:
            save(done)
            t0, tokens_since = time.time(), 0

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
