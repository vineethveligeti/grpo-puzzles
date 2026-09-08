#!/usr/bin/env python
"""Solve-rate evaluation on 500 held-out Countdown puzzles (same split every run).

    python eval_countdown.py --model Qwen/Qwen3.5-0.8B-Base            --tag base
    python eval_countdown.py --model runs/countdown-qwen3.5-0.8b-base/final --tag grpo400
    python eval_countdown.py --model Qwen/Qwen3.5-0.8B-Base --adapter runs/.../final --tag lora   # LoRA run

Greedy decoding by default (deterministic solve rate).  Add --sample --k 8 for pass@k.
Writes  results/<tag>.json  (metrics + 20 sample completions) so before/after can be plotted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_countdown
from rewards import answer_reward, extract_equation, format_reward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir to merge on top of --model")
    p.add_argument("--tag", required=True, help="name for results/<tag>.json")
    p.add_argument("--n", type=int, default=500, help="number of held-out puzzles (max 500)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--sample", action="store_true", help="temperature 1.0 sampling instead of greedy")
    p.add_argument("--k", type=int, default=1, help="samples per puzzle when --sample (pass@k)")
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, sample, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=1.0 if sample else None,
            top_p=1.0 if sample else None,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def main():
    a = parse_args()
    _, eval_ds = load_countdown(n_train=1)
    eval_ds = eval_ds.select(range(min(a.n, len(eval_ds))))

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()

    prompts = eval_ds["prompt"]
    targets, nums = eval_ds["target"], eval_ds["nums"]
    k = a.k if a.sample else 1

    t0 = time.time()
    solved_any = [0] * len(prompts)
    per_sample_solve, per_sample_format, lengths, samples = [], [], [], []
    for rep in range(k):
        comps = generate(model, tok, prompts, a.max_new_tokens, a.sample, a.batch_size)
        ar = answer_reward(comps, target=targets, nums=nums)
        fr = format_reward(comps)
        per_sample_solve.extend(ar)
        per_sample_format.extend(fr)
        lengths.extend(len(tok(c)["input_ids"]) for c in comps)
        for j, r in enumerate(ar):
            solved_any[j] = max(solved_any[j], int(r))
        if rep == 0:
            samples = [
                {"nums": n, "target": t, "completion": c, "equation": extract_equation(c), "solved": bool(r)}
                for n, t, c, r in list(zip(nums, targets, comps, ar))[:20]
            ]
    elapsed = time.time() - t0

    n3 = [i for i, n in enumerate(nums) if len(n) == 3]
    n4 = [i for i, n in enumerate(nums) if len(n) == 4]
    first = per_sample_solve[: len(prompts)]  # first sample per prompt
    metrics = {
        "tag": a.tag,
        "model": a.model,
        "adapter": a.adapter,
        "n_puzzles": len(prompts),
        "decoding": f"sample k={k}" if a.sample else "greedy",
        "solve_rate": statistics.mean(first),
        "solve_rate_3nums": statistics.mean(first[i] for i in n3) if n3 else None,
        "solve_rate_4nums": statistics.mean(first[i] for i in n4) if n4 else None,
        f"pass@{k}": statistics.mean(solved_any),
        "format_rate_clean": statistics.mean(1.0 if f == 1.0 else 0.0 for f in per_sample_format),
        "format_rate_any": statistics.mean(1.0 if f > 0 else 0.0 for f in per_sample_format),
        "mean_completion_tokens": statistics.mean(lengths),
        "seconds": round(elapsed, 1),
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as f:
        json.dump({"metrics": metrics, "samples": samples}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
