"""Regenerate results/curves_grpo800.png from the exported W&B history.

    python plot_curves.py            # reads results/wandb_history_stage1_and_1b.csv

Stage 1b was a resumed run; W&B logged one row at global_step 0 before the trainer restored
the step counter, which drew a spurious line from step 0. Rows of a resumed stage below its
start step are dropped here.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CSV = HERE / "results" / "wandb_history_stage1_and_1b.csv"
OUT = HERE / "results" / "curves_grpo800.png"
PANELS = [
    ("train/rewards/answer_reward/mean", "solve rate (train prompts, T=1)"),
    ("train/rewards/format_reward/mean", "format reward"),
    ("train/completions/mean_length", "completion length"),
    ("train/entropy", "entropy"),
]
WINDOW = 5  # rolling mean over logged rows (~4 optimizer steps apart)


def smooth(xs, w=WINDOW):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
    return out


rows = list(csv.DictReader(open(CSV)))
runs = {}
for r in rows:
    if not r.get("train/global_step"):
        continue
    runs.setdefault(r["run"], []).append(r)

fig, axes = plt.subplots(1, len(PANELS), figsize=(20, 4))
for ax, (key, title) in zip(axes, PANELS):
    for name, rs in runs.items():
        pts = [(float(r["train/global_step"]), float(r[key])) for r in rs if r.get(key)]
        pts.sort()
        start = min(s for s, _ in pts)
        # resumed stage: drop the pre-restore rows (a "400-800" stage keeps steps >= 400)
        if "400-800" in name:
            pts = [(s, v) for s, v in pts if s >= 400]
        xs = [s for s, _ in pts]
        ys = smooth([v for _, v in pts])
        ax.plot(xs, ys, label=name, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
