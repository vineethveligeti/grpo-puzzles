#!/usr/bin/env python
"""Per-check evaluation on a Wordmaze split — reproduces the blog post's table for YOUR model.

    python eval_wordmaze.py --model Qwen/Qwen3.5-4B --tag 4b-base --config m3-4 --split validation
    python eval_wordmaze.py --model Qwen/Qwen3.5-4B --adapter runs/wordmaze-qwen3.5-4b-lora/final --tag 4b-grpo
    python eval_wordmaze.py --model Qwen/Qwen3.5-4B --tag 4b-4x3 --word_lengths 4 --move_counts 3   # one bucket

Uses the LENIENT dictionary (any word wordfreq knows) so numbers are comparable with the published
baselines (Qwen3.5-4B 18.5% / Qwen3.5-9B 32.5% fully_valid on m3-4 test, thinking mode).
Sampling defaults follow Qwen's thinking-mode recommendation (T=0.6, top_p=0.95, top_k=20);
pass --greedy for deterministic decoding.  Writes results/<tag>.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_wordmaze
from verifier import CHECKS, Dictionary, extract_path, is_valid_ladder, verify


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--style", default="chat", choices=["chat", "base"])
    p.add_argument("--no_thinking", action="store_true")
    p.add_argument("--config", default="m3-4", choices=["m3-4", "m4-6"])
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--word_lengths", type=int, nargs="*", default=None)
    p.add_argument("--move_counts", type=int, nargs="*", default=None)
    p.add_argument("--spaced", action="store_true")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--dictionary", default="lenient", choices=["lenient", "top10k"])
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


@torch.no_grad()
def generate(model, tok, prompts, a):
    outs = []
    for i in range(0, len(prompts), a.batch_size):
        batch = prompts[i : i + a.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(do_sample=False) if a.greedy else dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        outs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"  generated {min(i + a.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def main():
    a = parse_args()
    thinking = not a.no_thinking
    think_open = a.style == "base" or thinking
    ds = load_wordmaze(a.config, a.split, style=a.style, use_spaced=a.spaced,
                       word_lengths=a.word_lengths, move_counts=a.move_counts)
    ds = ds.select(range(min(a.n, len(ds))))

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.style == "chat":
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                   for m in ds["prompt"]]
    else:
        prompts = list(ds["prompt"])

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval()

    t0 = time.time()
    comps = generate(model, tok, prompts, a)
    d = Dictionary(a.dictionary)
    checks = [verify(c, r["start"], r["goal"], r["word_length"], r["max_moves"], r["password"],
                     dictionary=d, think_open=think_open) for c, r in zip(comps, ds)]

    rates = {k: statistics.mean(1.0 if ch[k] else 0.0 for ch in checks) for k in CHECKS}
    rates["valid_ladder"] = statistics.mean(1.0 if is_valid_ladder(ch) else 0.0 for ch in checks)
    rates["answered"] = statistics.mean(1.0 if extract_path(c, think_open) else 0.0 for c in comps)
    rates["truncated_no_answer"] = statistics.mean(
        1.0 if ("</think>" not in c and think_open) else 0.0 for c in comps)
    by_bucket = collections.defaultdict(list)
    for ch, r in zip(checks, ds):
        by_bucket[f"len{r['word_length']}_moves{r['max_moves']}"].append(1.0 if ch["fully_valid"] else 0.0)
    metrics = {
        "tag": a.tag, "model": a.model, "adapter": a.adapter, "config": a.config, "split": a.split,
        "n": len(ds), "thinking": thinking, "spaced": a.spaced, "dictionary": a.dictionary,
        "decoding": "greedy" if a.greedy else "T0.6 top_p0.95 top_k20",
        "rates": rates,
        "fully_valid_by_bucket": {k: statistics.mean(v) for k, v in sorted(by_bucket.items())},
        "mean_completion_tokens": statistics.mean(len(tok(c)["input_ids"]) for c in comps),
        "seconds": round(time.time() - t0, 1),
    }
    samples = [{"puzzle": {k: r[k] for k in ("start", "goal", "word_length", "max_moves", "password")},
                "completion_tail": c[-600:], "path": extract_path(c, think_open), "fully_valid": ch["fully_valid"]}
               for c, r, ch in list(zip(comps, ds, checks))[:15]]
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as f:
        json.dump({"metrics": metrics, "samples": samples}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
