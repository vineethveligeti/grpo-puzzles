"""Wordmaze verifier: the 10 boolean checks from dipkumar.dev/posts/llm/wordmaze, as a pure function.

    from verifier import verify, Dictionary
    checks = verify("<answer>cold -> cord -> word -> ward -> warm</answer>",
                    start="cold", goal="warm", word_length=4, max_moves=4, password="FFBF")
    checks["fully_valid"]  # True

Check            | passes when
-----------------|------------------------------------------------
valid_format     | parses to >=2 alphabetic words of one common length
starts_correctly | first word == start
reaches_goal     | last word == goal
within_max_moves | steps <= max_moves
exact_moves      | steps == max_moves
one_letter_changes | every step changes exactly one letter
all_valid_words  | every word is in the dictionary
correct_word_length | every word has length word_length
password_matches | F/B string computed from the path == password
fully_valid      | all of the above

Dictionary modes
  "lenient" (author's grader): any word wordfreq has frequency data for  -> zipf_frequency(w,'en') > 0
                               NOTE: this accepts junk like 'zzzz' (zipf 1.8). Use for EVAL to be
                               comparable with the published baselines.
  "top10k"  (generator vocab): wordfreq top_n_list('en', 10000), a-z only. Every dataset puzzle is
                               solvable inside it (that's how puzzles were made). Use for TRAINING
                               so the policy can't be rewarded for junk 'words'.

Answers are read from the FIRST <answer>...</answer> that appears AFTER the last </think> (if any),
so text inside a thinking block never counts as the answer.  Spaces inside a word are removed
("c o l d" -> "cold") to support the spelled-out-letters ablation from the post.
"""

from __future__ import annotations

import re
from functools import lru_cache

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class Dictionary:
    def __init__(self, mode: str = "lenient", min_zipf: float = 0.0):
        assert mode in ("lenient", "top10k"), mode
        self.mode, self.min_zipf = mode, min_zipf
        self._top = None
        if mode == "top10k":
            from wordfreq import top_n_list
            self._top = {w for w in top_n_list("en", 10_000) if re.fullmatch(r"[a-z]+", w)}

    @lru_cache(maxsize=200_000)
    def __contains__(self, word: str) -> bool:
        if not re.fullmatch(r"[a-z]+", word):
            return False
        if self.mode == "top10k":
            return word in self._top
        from wordfreq import zipf_frequency
        z = zipf_frequency(word, "en")
        return z > 0 and z >= self.min_zipf


_DEFAULT_DICT = None


def default_dictionary() -> Dictionary:
    global _DEFAULT_DICT
    if _DEFAULT_DICT is None:
        _DEFAULT_DICT = Dictionary("lenient")
    return _DEFAULT_DICT


def extract_path(text: str, think_open: bool = False) -> list[str] | None:
    """Words of the answered path, or None if no <answer> tag (after the last </think>).

    think_open=True means the PROMPT already opened a <think> block (Qwen thinking mode, or the
    TinyZero-style base prompt).  Then a completion with no </think> is an unfinished thought and
    yields None — an <answer> written inside the thinking never counts.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif think_open or "<think>" in text:
        return None
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return None
    words = [w.strip().lower().replace(" ", "") for w in raw.split("->")]
    return words


def step_direction(a: str, b: str) -> str | None:
    """'F' / 'B' for a one-letter change a->b; None if not exactly one letter differs."""
    if len(a) != len(b):
        return None
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diffs) != 1:
        return None
    x, y = diffs[0]
    return "F" if y > x else "B"


CHECKS = [
    "valid_format", "starts_correctly", "reaches_goal", "within_max_moves", "exact_moves",
    "one_letter_changes", "all_valid_words", "correct_word_length", "password_matches", "fully_valid",
]


def verify(text: str, start: str, goal: str, word_length: int, max_moves: int, password: str,
           dictionary: Dictionary | None = None, think_open: bool = False) -> dict[str, bool]:
    d = dictionary or default_dictionary()
    start, goal, password = start.lower(), goal.lower(), password.upper()
    path = extract_path(text, think_open=think_open)
    r = {k: False for k in CHECKS}
    if not path or len(path) < 2 or not all(re.fullmatch(r"[a-z]+", w) for w in path) \
            or len({len(w) for w in path}) != 1:
        return r                                   # valid_format False -> everything False
    r["valid_format"] = True
    steps = len(path) - 1
    r["starts_correctly"] = path[0] == start
    r["reaches_goal"] = path[-1] == goal
    r["within_max_moves"] = steps <= max_moves
    r["exact_moves"] = steps == max_moves
    dirs = [step_direction(a, b) for a, b in zip(path, path[1:])]
    r["one_letter_changes"] = all(x is not None for x in dirs)
    r["all_valid_words"] = all(w in d for w in path)
    r["correct_word_length"] = all(len(w) == word_length for w in path)
    r["password_matches"] = r["one_letter_changes"] and "".join(dirs) == password
    r["fully_valid"] = all(r[k] for k in CHECKS if k != "fully_valid")
    return r


def is_valid_ladder(checks: dict[str, bool]) -> bool:
    """A real word ladder from start to goal in exactly max_moves steps — password ignored."""
    return all(checks[k] for k in ("valid_format", "starts_correctly", "reaches_goal", "exact_moves",
                                   "one_letter_changes", "all_valid_words", "correct_word_length"))


# ----------------------------------------------------------------------------
# Self-test:  python verifier.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    P = dict(start="cold", goal="warm", word_length=4, max_moves=4, password="FFBF")
    cases = {
        # from the blog post
        "canonical":        ("<answer>cold -> cord -> word -> ward -> warm</answer>", True),
        "inside_think_only": ("<think><answer>cold -> cord -> word -> ward -> warm</answer></think> no answer", False),
        "think_then_answer": ("<think>l->r F, c->w F...</think>\n<answer>cold -> cord -> word -> ward -> warm</answer>", True),
        "spaced_letters":   ("<answer>c o l d -> c o r d -> w o r d -> w a r d -> w a r m</answer>", True),
        "two_word_shortcut": ("<answer>cold -> warm</answer>", False),          # the hack from the post
        "alt_solution":     ("<answer>cold -> wold -> word -> ward -> warm</answer>", True),   # different path, same FFBF -> correct!
        "wrong_password":   ("<answer>cold -> cord -> card -> ward -> warm</answer>", False),  # F B F F != FFBF
        "too_many_moves":   ("<answer>cold -> cord -> card -> ward -> warm -> worm</answer>", False),
        "junk_word":        ("<answer>cold -> cqld -> cord -> word -> warm</answer>", False),
        "no_tag":           ("cold -> cord -> word -> ward -> warm", False),
        "unclosed_think":   ("<think>maybe <answer>cold -> cord -> word -> ward -> warm</answer> hmm", False),
        "uppercase":        ("<answer>Cold -> Cord -> Word -> Ward -> Warm</answer>", True),
    }
    ok = True
    for name, (txt, want) in cases.items():
        c = verify(txt, **P)
        flag = "OK " if c["fully_valid"] == want else "FAIL"
        ok &= flag == "OK "
        failed = [k for k, v in c.items() if not v and k != "fully_valid"]
        print(f"{flag} {name:18s} fully_valid={c['fully_valid']!s:5s} failing={failed}")

    # 4B failure example from the post: shark -> shore changes two letters
    c = verify("<answer>stark -> shark -> shore -> shire -> share</answer>",
               start="stark", goal="share", word_length=5, max_moves=4, password="BFBF")
    print("post 4B example: one_letter_changes =", c["one_letter_changes"], "(want False)")
    ok &= c["one_letter_changes"] is False

    # password direction sanity: cold->wold is c->w (F); wold->word l->r (F); word->ward o->a (B); ward->warm d->m (F)
    # think_open=True: prompt opened <think>; a completion without </think> is an unfinished thought
    c = verify("so the path is <answer>cold -> cord -> word -> ward -> warm</answer>", think_open=True, **P)
    print("think_open, no </think>: fully_valid =", c["fully_valid"], "(want False)"); ok &= not c["fully_valid"]
    c = verify("so the path is </think><answer>cold -> cord -> word -> ward -> warm</answer>", think_open=True, **P)
    print("think_open, closed     : fully_valid =", c["fully_valid"], "(want True)"); ok &= c["fully_valid"]
    print("dirs:", [step_direction(a, b) for a, b in zip(["cold", "wold", "word", "ward"], ["wold", "word", "ward", "warm"])])

    # strict dictionary blocks junk that lenient accepts
    strict, lenient = Dictionary("top10k"), Dictionary("lenient")
    print("'zzzz' lenient:", "zzzz" in lenient, "| top10k:", "zzzz" in strict, "(want True / False)")
    ok &= ("zzzz" in lenient) and ("zzzz" not in strict)
    print("dataset words in top10k:", all(w in strict for w in ["sleep", "sheep", "sheer", "cheer", "cheek", "pounds", "sounds", "wounds", "rounds"]))
    print("\nALL PASS" if ok else "\nSOME FAILED")
