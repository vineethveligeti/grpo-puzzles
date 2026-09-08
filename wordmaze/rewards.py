"""Reward functions for Wordmaze GRPO — designed to be hard to game.

The author of Wordmaze reports that a naive reward (e.g. "valid path that reaches the goal")
made a 4B model WORSE than its base: it learned the two-word shortcut
`<answer>cold -> warm</answer>`, which passes the loose checks and ignores the password.

Rule we follow:  only pay for things that are strictly on the road to a full solve, and make
the full solve dominate.

  solve_reward   1.0  fully_valid (all 10 checks)                <- the thing we want
  ladder_reward  0.3  a REAL ladder start->goal in exactly max_moves steps, every step one
                      letter, every word in the dictionary — password ignored.  You cannot get
                      this by bluffing; it is the sub-skill "track letters + search word graph".
  format_reward  0.1  an <answer> tag that parses to >=2 words (nudges "always answer";
                      near-free, so it contributes ~zero advantage once learned — harmless)

  max total = 1.4.   The two-word hack scores 0.1.  A valid ladder with the wrong password
  scores 0.4.  A correct solve scores 1.4.  Gradient always points toward the real objective.

Training dictionary is the generator's own top-10k vocabulary ("top10k") so junk words that the
lenient wordfreq list happens to know (e.g. 'zzzz') never earn ladder credit.  Evaluate with the
lenient dictionary (eval_wordmaze.py) to stay comparable with the published baselines.

TRL calls  fn(prompts, completions, **dataset_columns); columns here:
  start, goal, word_length, max_moves, password  (all lists, one per completion)
"""

from __future__ import annotations

from verifier import Dictionary, is_valid_ladder, verify

_TRAIN_DICT = None
THINK_OPEN = True   # our prompts (chat thinking-mode or base template) open a <think> block; see verifier.extract_path


def set_think_open(flag: bool) -> None:
    global THINK_OPEN
    THINK_OPEN = flag


def _train_dict() -> Dictionary:
    global _TRAIN_DICT
    if _TRAIN_DICT is None:
        _TRAIN_DICT = Dictionary("top10k")
    return _TRAIN_DICT


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):   # conversational format
        return c[0].get("content", "")
    return str(c)


def _checks(completions, start, goal, word_length, max_moves, password):
    d = _train_dict()
    return [
        verify(_text(c), s, g, int(wl), int(mm), pw, dictionary=d, think_open=THINK_OPEN)
        for c, s, g, wl, mm, pw in zip(completions, start, goal, word_length, max_moves, password)
    ]


def solve_reward(completions, start, goal, word_length, max_moves, password, **kwargs) -> list[float]:
    return [1.0 if ch["fully_valid"] else 0.0
            for ch in _checks(completions, start, goal, word_length, max_moves, password)]


def ladder_reward(completions, start, goal, word_length, max_moves, password, **kwargs) -> list[float]:
    return [0.3 if is_valid_ladder(ch) else 0.0
            for ch in _checks(completions, start, goal, word_length, max_moves, password)]


def format_reward(completions, start, goal, word_length, max_moves, password, **kwargs) -> list[float]:
    return [0.1 if ch["valid_format"] else 0.0
            for ch in _checks(completions, start, goal, word_length, max_moves, password)]


REWARD_FUNCS = [format_reward, ladder_reward, solve_reward]


if __name__ == "__main__":
    kw = dict(start=["cold"] * 4, goal=["warm"] * 4, word_length=[4] * 4, max_moves=[4] * 4, password=["FFBF"] * 4)
    comps = [
        "<think>...</think><answer>cold -> cord -> word -> ward -> warm</answer>",   # solve
        "<think>...</think><answer>cold -> cord -> card -> ward -> warm</answer>",   # ladder, wrong password
        "<think>...</think><answer>cold -> warm</answer>",                           # the hack
        "<think>I give up",                                                          # nothing
    ]
    tot = [sum(x) for x in zip(format_reward(comps, **kw), ladder_reward(comps, **kw), solve_reward(comps, **kw))]
    for c, t in zip(comps, tot):
        print(f"{t:4.1f}  {c}")
    assert tot == [1.4, 0.4, 0.1, 0.0], tot
    print("ALL PASS")
