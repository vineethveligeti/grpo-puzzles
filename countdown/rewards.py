"""Rule-based rewards for the Countdown task (TRL GRPOTrainer signature).

TRL calls each reward function as  fn(prompts, completions, **dataset_columns)
where every non-`prompt` column of the dataset arrives as a keyword list
(here: `target` and `nums`).  Each function returns one float per completion.

Two separate rewards so that TRL logs them separately
(`rewards/format_reward/mean`, `rewards/answer_reward/mean`):

  format_reward : 1.0  -> "...</think>\n<answer>EQ</answer>" and NOTHING after
                  0.5  -> tags present in the right order but junk after </answer>
                  0.0  -> otherwise
  answer_reward : 1.0  -> equation uses every given number exactly once,
                          only + - * / ( ), and evaluates to target
                  0.0  -> otherwise

Total per completion is therefore in [0, 2].  The metric you care about is
`rewards/answer_reward/mean` == fraction of completions that solve the puzzle.

Reference implementations this is distilled from:
  * TinyZero  verl/utils/reward_score/countdown.py  (score 1.0 / format 0.1 / 0)
  * Phil Schmid Mini-R1 notebook                    (format 0/1 + equation 0/1)
"""

from __future__ import annotations

import re

# The prompt ends with "<think>", so the completion starts INSIDE the think block.
_FORMAT_RE = re.compile(
    r"^(?:(?!</think>).)*?</think>\s*<answer>(?:(?!</answer>).)*?</answer>(?P<tail>.*)$",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ALLOWED_EQ = re.compile(r"^[\d+\-*/().\s]+$")


def _completion_text(c) -> str:
    """Accept both TRL formats: plain string (standard) or [{'role','content'}] (conversational)."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content", "")
    return str(c)


def extract_equation(text: str) -> str | None:
    """First <answer>...</answer> block, stripped. None if absent."""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def equation_is_correct(equation: str | None, nums: list[int], target: int) -> bool:
    if not equation:
        return False
    if not _ALLOWED_EQ.match(equation):          # letters, names, etc. -> reject
        return False
    if "**" in equation or len(equation) > 200:  # '**' passes the char filter; 9**9**9 would hang eval
        return False
    used = [int(n) for n in re.findall(r"\d+", equation)]
    if sorted(used) != sorted(int(n) for n in nums):  # each number exactly once
        return False
    try:
        # Characters are restricted to digits/operators/parens/dot/space above,
        # so eval cannot reference any name.  Division by zero -> exception -> False.
        result = eval(equation, {"__builtins__": None}, {})
    except Exception:
        return False
    try:
        return abs(float(result) - float(target)) < 1e-6
    except Exception:
        return False


def format_reward(completions, **kwargs) -> list[float]:
    out = []
    for c in completions:
        text = _completion_text(c)
        m = _FORMAT_RE.match(text)
        if m is None:
            out.append(0.0)
        elif m.group("tail").strip() == "":
            out.append(1.0)
        else:
            out.append(0.5)
    return out


def answer_reward(completions, target, nums, **kwargs) -> list[float]:
    out = []
    for c, t, n in zip(completions, target, nums):
        eq = extract_equation(_completion_text(c))
        out.append(1.0 if equation_is_correct(eq, n, t) else 0.0)
    return out


# ----------------------------------------------------------------------------
# Self-test:  python rewards.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    nums, target = [44, 19, 35], 98

    good = " 44+19 = 63, 63+35 = 98. </think>\n<answer> 44 + 19 + 35 </answer>"
    junk_after = good + "\nUser: another question"
    wrong_value = " ... </think>\n<answer> 44 + 19 - 35 </answer>"
    reuse_number = " ... </think>\n<answer> 44 + 19 + 35 + 0 </answer>"        # 0 not given
    twice = " ... </think>\n<answer> 44 + 44 + 10 </answer>"                 # 44 twice
    missing_num = " ... </think>\n<answer> 63 + 35 </answer>"                 # 63 not given
    no_think_close = " 44+19+35 <answer> 44 + 19 + 35 </answer>"             # no </think>
    code_inject = " ... </think>\n<answer> __import__('os') </answer>"
    power = " ... </think>\n<answer> 44 ** 19 </answer>"                       # '**' passes regex? -> chars ok, eval huge
    division = " ... </think>\n<answer> (44 * 35) / 19 </answer>"             # 81.05 != 98
    zero_div = " ... </think>\n<answer> 44 / (19 - 19) </answer>"             # 19 twice anyway

    cases = {
        "good": (good, 1.0, 1.0),
        "junk_after": (junk_after, 0.5, 1.0),
        "wrong_value": (wrong_value, 1.0, 0.0),
        "reuse_number": (reuse_number, 1.0, 0.0),
        "twice": (twice, 1.0, 0.0),
        "missing_num": (missing_num, 1.0, 0.0),
        "no_think_close": (no_think_close, 0.0, 1.0),
        "code_inject": (code_inject, 1.0, 0.0),
        "power": (power, 1.0, 0.0),
        "division": (division, 1.0, 0.0),
        "zero_div": (zero_div, 1.0, 0.0),
    }
    comps = [v[0] for v in cases.values()]
    f = format_reward(comps)
    a = answer_reward(comps, target=[target] * len(comps), nums=[nums] * len(comps))
    ok = True
    for (name, (_, ef, ea)), gf, ga in zip(cases.items(), f, a):
        flag = "OK " if (gf == ef and ga == ea) else "FAIL"
        ok &= flag == "OK "
        print(f"{flag} {name:16s} format={gf} (want {ef})  answer={ga} (want {ea})")
    # 4-number case with division that must be exact
    print("4-num exact division:",
          answer_reward(["</think><answer>(100 / 25) * (3 + 6)</answer>"], target=[36], nums=[[100, 25, 3, 6]]))
    print("\nALL PASS" if ok else "\nSOME FAILED")
