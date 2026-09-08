"""Dataset + prompt construction for Countdown GRPO.

Source: Jiayi-Pan/Countdown-Tasks-3to4  (490,364 rows; columns `target` int, `nums` list[int])
  * 240,632 rows have 3 numbers, 249,732 have 4 numbers
  * targets are 10..100, operands 1..99

We use the TinyZero *base-model* template verbatim (no chat template, no system role).
The prompt ends with "<think>" so the model's completion starts inside the think block.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
N_EVAL = 500          # held-out puzzles, never seen in training (taken from the shuffled head)
SEED = 42

# NOTE: identical to TinyZero examples/data_preprocess/countdown.py, template_type='base'
PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
    "User: Using the numbers {nums}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
    "for example <answer> (1 + 2) / 3 </answer>.\n"
    "Assistant: Let me solve this step by step.\n"
    "<think>"
)


def build_prompt(nums: list[int], target: int) -> str:
    return PROMPT_TEMPLATE.format(nums=list(nums), target=target)


def _to_prompt_row(ex):
    return {"prompt": build_prompt(ex["nums"], ex["target"]), "target": ex["target"], "nums": ex["nums"]}


def load_countdown(n_train: int = 20_000, curriculum: str = "none", seed: int = SEED):
    """Return (train_ds, eval_ds) with columns prompt / target / nums.

    curriculum:
      "none"   -> mix of 3- and 4-number puzzles (the raw distribution)
      "3only"  -> only 3-number puzzles (much easier; use this if the base model shows ~0% solve rate)
      "4only"  -> only 4-number puzzles (harder second stage)
    The eval split is always the raw mix so numbers are comparable across runs.
    """
    ds = load_dataset(DATASET_ID, split="train").shuffle(seed=seed)
    eval_ds = ds.select(range(N_EVAL))
    pool = ds.select(range(N_EVAL, len(ds)))
    if curriculum == "3only":
        pool = pool.filter(lambda ex: len(ex["nums"]) == 3)
    elif curriculum == "4only":
        pool = pool.filter(lambda ex: len(ex["nums"]) == 4)
    elif curriculum != "none":
        raise ValueError(f"unknown curriculum {curriculum!r}")
    train_ds = pool.select(range(min(n_train, len(pool))))
    return train_ds.map(_to_prompt_row), eval_ds.map(_to_prompt_row)


if __name__ == "__main__":
    tr, ev = load_countdown(n_train=100, curriculum="3only")
    print(tr)
    print(ev)
    print("---- example prompt ----")
    print(tr[0]["prompt"])
    print("---- target / nums ----", tr[0]["target"], tr[0]["nums"])
