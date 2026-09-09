#!/usr/bin/env python
"""Regenerate countdown_grpo_colab.ipynb as a SELF-CONTAINED notebook.

The four .py files in this folder are embedded as %%writefile cells, so the notebook needs no
git clone / upload step.  Re-run this after editing any of them:

    python build_colab_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["data.py", "rewards.py", "train_countdown_grpo.py", "eval_countdown.py"]
WORK = "/content/grpo_puzzles/countdown"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = [
    md("""# Countdown × GRPO — Qwen3.5-0.8B-Base, no SFT

Runner for `countdown/GUIDE.md` (the "48+63=111" TinyZero recipe on a 2026 0.8B base model).
Self-contained: the four `.py` files are written to disk by cells below — nothing to clone.

**Runtime → Change runtime type → A100** (40 GB). L4 works with the smaller batch noted in the train cell.

Order: install → self-tests → **baseline eval (go/no-go, ~10 min)** → 5-step timing run → 400-step train → curves → after-eval."""),

    code("""!nvidia-smi
import torch; print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")"""),

    code("""%%capture
!pip install -U "trl>=1.11" "transformers>=5.2" datasets peft accelerate wandb matplotlib
!pip install flash-linear-attention          # Qwen3.5 Gated-DeltaNet kernels (Triton). Without it: silent slow path.
# Optional & minor (often fails to build on Colab; skip if so): !pip install causal-conv1d --no-build-isolation
# Optional fast generation:     !pip install "vllm==0.27.1"   -> then add --use_vllm to the train command"""),

    code("""# Must print "fla OK" — transformers 5.16 gives NO warning when the fast kernels are missing.
!python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla OK')"
import trl, transformers, datasets; print("trl", trl.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)"""),

    md("## Code (embedded copies of `countdown/*.py`; regenerate with `build_colab_notebook.py` after edits)"),
    code(f"!mkdir -p {WORK}\n%cd {WORK}"),
]

for name in FILES:
    src = (HERE / name).read_text()
    cells.append(code(f"%%writefile {WORK}/{name}\n{src}"))

cells += [
    code("""!python rewards.py | tail -3
!python data.py | head -5"""),

    md("""## Optional: tokens via Colab Secrets (🔑 icon in the left sidebar)
Add `HF_TOKEN` and/or `WANDB_API_KEY` there and grant this notebook access. Nothing below *requires* them:
the baseline eval runs without any login; W&B just gives you the curves as a shareable link."""),

    code("""import os
HF_OK = WANDB_OK = False
try:
    from google.colab import userdata
    for k in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            v = userdata.get(k)
            if v: os.environ[k] = v
        except Exception:
            pass
except ImportError:
    pass
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login; login(token=os.environ["HF_TOKEN"]); HF_OK = True
if os.environ.get("WANDB_API_KEY"):
    import wandb; wandb.login(key=os.environ["WANDB_API_KEY"]); WANDB_OK = True
REPORT_TO = "wandb" if WANDB_OK else "none"
print(f"HF login: {HF_OK} | W&B: {WANDB_OK} -> report_to={REPORT_TO}")"""),

    md("""## Step 3 — baseline (go / no-go). ~10 min on an A100.
Greedy solve rate on 500 held-out puzzles, then pass@8 on 200 of them. The decision table is GUIDE.md §5."""),

    code("""!python eval_countdown.py --model Qwen/Qwen3.5-0.8B-Base --tag base-greedy
!python eval_countdown.py --model Qwen/Qwen3.5-0.8B-Base --tag base-pass8 --sample --k 8 --n 200"""),

    code("""import json
g = json.load(open("results/base-greedy.json"))["metrics"]
p = json.load(open("results/base-pass8.json"))["metrics"]
print(f"greedy : solve={g['solve_rate']:.3f}  3nums={g['solve_rate_3nums']:.3f}  4nums={g['solve_rate_4nums']:.3f}  clean_format={g['format_rate_clean']:.3f}  any_format={g['format_rate_any']:.3f}  len={g['mean_completion_tokens']:.0f}")
print(f"pass@8 : {p['pass@8']:.3f}   (first-sample solve={p['solve_rate']:.3f}, 3nums={p['solve_rate_3nums']:.3f})")
pk = p["pass@8"]
if pk >= 0.05:   print("\\n=> GO: train on the mix        (CURRICULUM = 'none')")
elif pk >= 0.01: print("\\n=> THIN: 3-number puzzles first (CURRICULUM = '3only' for ~150 steps, then 'none')")
else:            print("\\n=> NO SIGNAL at 0.8B-Base: try Qwen/Qwen3.5-2B-Base or Qwen/Qwen3.5-0.8B (instruct) — GUIDE.md §5")
print("\\n--- 3 base-model completions ---")
for s in json.load(open("results/base-greedy.json"))["samples"][:3]:
    print(s["nums"], "->", s["target"], "| eq:", s["equation"], "| solved:", s["solved"]); print(s["completion"][:400]); print("-"*80)"""),

    md("""## Step 4a — 5-step timing run (~5 min). Tells you the real seconds/step before you commit to 400.
Also check `frac_reward_zero_std` in the log: it must be clearly below 1.0 or GRPO has no gradient."""),

    code("""import time; t0 = time.time()
CURRICULUM = "3only"     # from the decision above: "none" | "3only"   (2026-09-08 baseline: pass@8 = 2.5% -> 3only)
MAXLEN = 1024            # 512 clipped 98% of base completions, 1024 still clips ~90%; terminated ones average ~680 tokens
# batch 8 x 1024 tokens OOMs on the A100-40GB in the fla backward -> 4 x 8 (same 32 completions/step)
!python train_countdown_grpo.py --output_dir runs/timing --max_steps 5 --save_steps 1000 --report_to none --curriculum $CURRICULUM --max_completion_length $MAXLEN --per_device_train_batch_size 4 --grad_accum 8
print(f"{(time.time()-t0)/5:.0f} s/step incl. model load; 400 steps <= {(time.time()-t0)/5*400/3600:.1f} h")"""),

    md("""## Step 4b — train (a few hours). First 5 steps in the log: `frac_reward_zero_std`, `rewards/answer_reward/mean`.
Batch 4 × accum 8 is already the A100 setting at 1024 tokens; on an L4 use `--lora --lr 2e-5` or `--max_completion_length 512`."""),

    code("""import os; os.environ["WANDB_PROJECT"] = "grpo-countdown"
RUN_DIR = "runs/countdown-qwen3.5-0.8b-base"
!python train_countdown_grpo.py --output_dir $RUN_DIR --max_steps 400 --report_to $REPORT_TO --run_name cd-0.8b-base-3only-len1024 --curriculum $CURRICULUM --max_completion_length $MAXLEN --per_device_train_batch_size 4 --grad_accum 8 --save_steps 50"""),

    md("## Step 5 — curves (from the checkpoint's `trainer_state.json`; W&B has the full set)"),

    code("""KEYS = ["rewards/answer_reward/mean", "rewards/format_reward/mean", "completions/mean_length", "frac_reward_zero_std"]
TITLES = ["solve rate (train prompts)", "format reward", "completion length", "dead groups (all-same reward)"]
import json, glob, os
import matplotlib.pyplot as plt

def latest_state(run_dir):
    ck = sorted(glob.glob(f"{run_dir}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    path = os.path.join(ck[-1], "trainer_state.json") if ck else os.path.join(run_dir, "final", "trainer_state.json")
    return json.load(open(path))["log_history"]

hist = latest_state(RUN_DIR)
def series(key):
    return [(h["step"], h[key]) for h in hist if key in h]

fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
for ax, key, title in zip(axes, KEYS, TITLES):
    s = series(key)
    if s:
        ax.plot(*zip(*s)); ax.set_title(title); ax.set_xlabel("optimizer step"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{RUN_DIR}/curves.png", dpi=150); plt.show()
print("saved", f"{RUN_DIR}/curves.png")"""),

    md("## Step 6 — after: same 500 held-out puzzles, before vs after"),

    code("""!python eval_countdown.py --model $RUN_DIR/final --tag grpo400
!python eval_countdown.py --model $RUN_DIR/final --tag grpo400-pass8 --sample --k 8 --n 200
import json
for tag in ["base-greedy", "grpo400"]:
    m = json.load(open(f"results/{tag}.json"))["metrics"]
    print(f"{tag:12s} solve={m['solve_rate']:.3f}  3nums={m['solve_rate_3nums']:.3f}  4nums={m['solve_rate_4nums']:.3f}  clean_format={m['format_rate_clean']:.3f}  len={m['mean_completion_tokens']:.0f}")"""),

    code("""# look at what it learned to say
import json
for s in json.load(open("results/grpo400.json"))["samples"][:5]:
    print(s["nums"], "->", s["target"], "| solved:", s["solved"]); print(s["completion"][:600]); print("-"*80)"""),

    md("""## Keep the outputs (Colab disks vanish on disconnect)
Either mount Drive and copy `results/` + `curves.png` + `final/`, or download them from the file browser."""),

    code("""# from google.colab import drive; drive.mount('/content/drive')
# !mkdir -p /content/drive/MyDrive/grpo_puzzles && cp -r results $RUN_DIR/curves.png /content/drive/MyDrive/grpo_puzzles/
# Publish the model (needs HF_TOKEN secret):
# from transformers import AutoModelForCausalLM, AutoTokenizer
# AutoModelForCausalLM.from_pretrained(f"{RUN_DIR}/final").push_to_hub("<you>/qwen3.5-0.8b-countdown-grpo")
# AutoTokenizer.from_pretrained(f"{RUN_DIR}/final").push_to_hub("<you>/qwen3.5-0.8b-countdown-grpo")"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = HERE / "countdown_grpo_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells, {out.stat().st_size/1024:.0f} KB)")
