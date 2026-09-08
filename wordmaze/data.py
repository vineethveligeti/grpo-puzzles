"""Dataset + prompt construction for Wordmaze GRPO.

Source: immortal3/wordmaze  (2 configs x 2,000 puzzles; train 1,600 / validation 200 / test 200)
  m3-4 : 4-6 letter words, 3-4 moves   (easier — start here)
  m4-6 : 4-6 letter words, 4-6 moves   (the default/hard config in the blog post)
Columns: id, start, goal, word_length, max_moves, password, prompt, messages, solution, answer, path, num_moves

Prompt styles
  chat  : the dataset's own `messages` (system rules + user puzzle)  -> TRL applies the chat template.
          Use with an INSTRUCT model (Qwen/Qwen3.5-4B, -2B ...).  Thinking mode is switched on via
          GRPOConfig(chat_template_kwargs={"enable_thinking": True}) in the train script.
  base  : a plain-text TinyZero-style prompt ending in "<think>" for a BASE model.

--spaced: spell every word with spaces ("c o l d") in the prompt — the tokenization ablation from
          the blog post (+8pp for the 4B).  The verifier removes spaces inside words, so answers
          may come back spaced or not.
"""

from __future__ import annotations

import re

from datasets import load_dataset

DATASET_ID = "immortal3/wordmaze"

BASE_TEMPLATE = (
    "A conversation between User and Assistant. The user gives a word-ladder puzzle and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
    "User: Solve this Wordmaze puzzle. Rules: change exactly one letter per move; every word must be a valid "
    "English word of length {word_length}; reach the goal in exactly {max_moves} moves; for each move write F if "
    "the changed letter moves forward in the alphabet and B if backward — the F/B string must equal the password.\n"
    "start: {start}\ngoal: {goal}\nword_length: {word_length}\nmax_moves: {max_moves}\npassword: {password}\n"
    "Show your work in <think> </think> tags and give only the path in <answer> </answer> tags, "
    "for example <answer>cold -> cord -> word -> ward -> warm</answer>.\n"
    "Assistant: Let me solve this step by step.\n<think>"
)


def spaced(word: str) -> str:
    return " ".join(word)


def _space_words_in_text(text: str) -> str:
    """Replace `start: cold` / `goal: warm` values and the example ladder with spaced letters."""
    text = re.sub(r"(start|goal): ([a-z]+)", lambda m: f"{m.group(1)}: {spaced(m.group(2))}", text)
    text = re.sub(r"<answer>([a-z][a-z >-]*?)</answer>",
                  lambda m: "<answer>" + " -> ".join(spaced(w.strip()) for w in m.group(1).split("->")) + "</answer>",
                  text)
    return text


def _to_row(ex, style: str, use_spaced: bool):
    if style == "chat":
        msgs = [dict(m) for m in ex["messages"]]
        if use_spaced:
            msgs = [{**m, "content": _space_words_in_text(m["content"])} for m in msgs]
        prompt = msgs
    elif style == "base":
        prompt = BASE_TEMPLATE.format(start=ex["start"], goal=ex["goal"], word_length=ex["word_length"],
                                      max_moves=ex["max_moves"], password=ex["password"])
        if use_spaced:
            prompt = _space_words_in_text(prompt)
    else:
        raise ValueError(style)
    return {"prompt": prompt, "start": ex["start"], "goal": ex["goal"], "word_length": ex["word_length"],
            "max_moves": ex["max_moves"], "password": ex["password"]}


def load_wordmaze(config: str = "m3-4", split: str = "train", style: str = "chat", use_spaced: bool = False,
                  word_lengths: list[int] | None = None, move_counts: list[int] | None = None, seed: int = 42):
    """Return a Dataset with columns prompt/start/goal/word_length/max_moves/password.

    word_lengths / move_counts filter for a curriculum, e.g. word_lengths=[4], move_counts=[3]
    (the easiest bucket: 271 train puzzles in m3-4; each (length, moves) bucket has ~260-270).
    """
    ds = load_dataset(DATASET_ID, config, split=split)
    if word_lengths:
        ds = ds.filter(lambda ex: ex["word_length"] in word_lengths)
    if move_counts:
        ds = ds.filter(lambda ex: ex["max_moves"] in move_counts)
    ds = ds.shuffle(seed=seed)
    keep = {"prompt", "start", "goal", "word_length", "max_moves", "password"}
    ds = ds.map(lambda ex: _to_row(ex, style, use_spaced), remove_columns=[c for c in ds.column_names if c not in keep])
    return ds


if __name__ == "__main__":
    ds = load_wordmaze("m3-4", "train", style="chat", word_lengths=[4], move_counts=[3])
    print(ds)
    print(ds[0]["prompt"][1]["content"])
    ds2 = load_wordmaze("m3-4", "validation", style="base", use_spaced=True)
    print(ds2[0]["prompt"])
    import collections
    full = load_dataset(DATASET_ID, "m3-4", split="train")
    print("m3-4 train buckets (word_length, max_moves):",
          sorted(collections.Counter(zip(full["word_length"], full["max_moves"])).items()))
