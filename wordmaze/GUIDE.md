# Wordmaze × GRPO — step by step (the "try this instead" from the replies)

> **TL;DR.** Wordmaze is a one-shot word-ladder puzzle with a twist: the *directions* of the letter changes
> (F = forward in the alphabet, B = backward) must spell a given password. A pure-Python verifier grades it with
> 10 booleans. Frontier models solve it (Gemini 3.1 Pro 99.5 %); small open models don't (Qwen3.5-4B 18.5 % on
> the easy config, 2.5 % on the hard one). The author built it *for* GRPO on small models — and reports that his
> first naive GRPO run made the model **worse** by gaming the checks. He promised a "part two" training post; as of
> 2026-09-07 it has not appeared. **That gap is your project:** a reward that can't be gamed, a baseline-first
> protocol, a curve, and an honest write-up. Budget 1–2 weeks and treat the first week as measurement.

Read `../countdown/GUIDE.md` first if you haven't run a GRPO job yet — same stack, easier task.

---

## 0. The puzzle in 60 seconds

```
start: cold      goal: warm      word_length: 4      max_moves: 4      password: FFBF

cold -> cord -> word -> ward -> warm
     F       F       B       F          l→r forward, c→w forward, o→a backward, d→m forward
```

Rules: change exactly one letter per move; every word must be a real English word of that length; reach the
goal in exactly `max_moves` moves (the password has one letter per move, so its length *is* the move count);
the F/B string must equal the password. Answer format: `<answer>cold -> cord -> word -> ward -> warm</answer>`.

**Why small models fail** (the post's "tokenization barrier"): `shots` and `steps` are each one token; the model
can't *see* that three letters changed, it has to spell both words out in its head and compare. Spelling the words
letter-by-letter in the prompt (`c o l d`) lifted the 4B from 18.5 % → 26.5 %. That's an ablation you get for free
here (`--spaced`).

## 1. Ingredients (verified)

| piece | fact | source |
|---|---|---|
| Dataset | `immortal3/wordmaze`, configs `m3-4` (3–4 moves) and `m4-6` (4–6 moves), 4–6-letter words, each 1,600 train / 200 val / 200 test, ~270 puzzles per (length, moves) bucket | HF dataset card |
| Row fields | `start goal word_length max_moves password prompt messages solution answer path num_moves` — `messages` is a ready system+user chat prompt | dataset schema |
| Generation | wordfreq top-10k English words minus proper nouns → one-letter-neighbour graph → random simple walks → read (start, goal, password) off the walk ⇒ every puzzle solvable *within the top-10k vocabulary* | `create_dataset.py` |
| Grader | 10 checks (table below); `fully_valid` is the metric; the author's grader accepts any word wordfreq has data for (lenient) | blog post |
| Baselines (`fully_valid`, thinking mode) | m3-4 / m4-6: Gemini 3.1 Pro 99.5/99.5, Gemini 3.5 Flash 95.5/83.5, Gemini 2.5 Flash 56.5/36.5, **Qwen3.5-9B 32.5/9.5, Qwen3.5-4B 18.5/2.5** (9B m3-4 is n=40) | blog post |
| Where failures concentrate | `one_letter_changes` and `password_matches`; format and word validity are "nearly free" | blog post |
| The hack | reward "valid path that reaches the goal" → model emits `<answer>cold -> warm</answer>` | blog post ("More on that in the training post") |

## 2. The verifier (`verifier.py`) — 10 checks, 2 dictionaries

| check | passes when | free or hard? |
|---|---|---|
| `valid_format` | ≥2 same-length alphabetic words parsed from `<answer>` | free |
| `starts_correctly` / `reaches_goal` | endpoints match | free |
| `within_max_moves` | steps ≤ max | free |
| `exact_moves` | steps == max | easy |
| `correct_word_length` | all words right length | easy |
| `all_valid_words` | every word in the dictionary | medium |
| **`one_letter_changes`** | each step changes exactly one letter | **hard** (character-level) |
| **`password_matches`** | computed F/B == password | **hard** (character-level + planning) |
| `fully_valid` | all of the above | the metric |

Two things our implementation adds on top of the post:

1. **Two dictionaries.** `lenient` = wordfreq knows the word (zipf > 0) — matches the author's grader, but it
   accepts junk like `zzzz` (zipf 1.8), which a policy under RL *will* find. `top10k` = the generator's own
   vocabulary, in which every puzzle is solvable by construction. **Train with `top10k`, evaluate with `lenient`**
   so your numbers stay comparable to the published table.
2. **Thinking-aware extraction.** Only the first `<answer>` *after the last* `</think>` counts. If the prompt opened
   a `<think>` block (Qwen thinking mode or a base-model prompt) and the completion never closes it, that's an
   unfinished thought → no answer. Otherwise a model learns to drop `<answer>` tags inside its thinking.

`python verifier.py` runs 12 cases including the post's two-word hack and its `shark -> shore` two-letter example.

## 3. The reward (`rewards.py`) — designed against the known failure

```
solve_reward   = 1.0 if fully_valid
ladder_reward  = 0.3 if a REAL ladder start→goal, exact moves, one letter per step, all words in top-10k (password ignored)
format_reward  = 0.1 if an <answer> with ≥2 words was produced
```

| completion | total | note |
|---|---|---|
| correct solution | **1.4** | |
| valid ladder, wrong password | 0.4 | real sub-skill, cannot be bluffed |
| `<answer>cold -> warm</answer>` (the hack) | 0.1 | fails `exact_moves`, `one_letter_changes`, `password_matches` |
| junk words that wordfreq happens to know | 0.1 | `top10k` dictionary during training |
| thinking that never ends / no answer | 0.0 | |

Design rule: pay only for things strictly on the road to a full solve, and let the solve dominate. Everything
"free" gets the same 0.1 in every completion of a group → zero advantage → harmless. If you ever see
`rewards/ladder_reward/mean` rise while `rewards/solve_reward/mean` stays flat for 100+ steps, the model is
farming ladders; lower the ladder reward to 0.15 or drop it.

## 4. Step 0 — Pre-flight

Same as Countdown (README "Shared setup") plus `pip install wordfreq`. Self-tests: `python verifier.py`,
`python rewards.py`, `python data.py` (prints the bucket counts and an example prompt).

## 5. Step 1 — Pick the model with data, not vibes (this *is* the first result)

GRPO amplifies successes that already exist; it cannot conjure them. So measure `fully_valid` on the easy bucket
for candidate models **before** training. Each eval below is 200 puzzles × up to 2,048 tokens of thinking —
expect 10–30 min per model on an A100 with HF `generate`.

```bash
# the easiest bucket: 4-letter words, 3 moves (271 train puzzles in m3-4)
python eval_wordmaze.py --model Qwen/Qwen3.5-4B   --tag 4b-base-4x3 --word_lengths 4 --move_counts 3
python eval_wordmaze.py --model Qwen/Qwen3.5-2B   --tag 2b-base-4x3 --word_lengths 4 --move_counts 3
python eval_wordmaze.py --model Qwen/Qwen3.5-0.8B --tag 08b-base-4x3 --word_lengths 4 --move_counts 3
# the ablation the post suggests
python eval_wordmaze.py --model Qwen/Qwen3.5-4B   --tag 4b-base-4x3-spaced --word_lengths 4 --move_counts 3 --spaced
# full easy config, to compare with the published 18.5 % (that was the TEST split)
python eval_wordmaze.py --model Qwen/Qwen3.5-4B   --tag 4b-base-m34 --split test
```

Decision table (on the 4×3 bucket, `rates.fully_valid`):

| result | do |
|---|---|
| 4B ≥ 10 % | train 4B on the 4×3 bucket (default command) |
| 2B ≥ 5 % | train 2B — 2× faster steps, more of them; 4B as the "scale" ablation later |
| everything < 3 % but `valid_ladder` > 10 % | the ladder reward will carry the early phase; still train, expect slow start |
| everything ≈ 0 and `truncated_no_answer` high | the model thinks past 2,048 tokens; raise `--max_new_tokens 4096` for eval, and consider `--spaced` |
| everything ≈ 0 even with `--spaced` | write *that* up (it's a real negative result), then do the Countdown project instead |

This table of baselines — 3 model sizes × {plain, spaced} × per-check rates — is already a publishable figure
and directly extends the author's table with 0.8B/2B rows he didn't run.

## 6. Step 2 — Train

```bash
# default: Qwen3.5-4B instruct, thinking on, LoRA r=32, m3-4, 4-letter/3-move bucket, 200 steps
python train_wordmaze_grpo.py --report_to wandb --run_name wm-4b-4x3

# curriculum: widen once solve rate on 4x3 is > 40 %
python train_wordmaze_grpo.py --word_lengths 4 5 --move_counts 3 4 --resume runs/wordmaze-qwen3.5-4b-lora/checkpoint-200

# ablations
python train_wordmaze_grpo.py --spaced --run_name wm-4b-4x3-spaced
python train_wordmaze_grpo.py --model Qwen/Qwen3.5-2B --output_dir runs/wm-2b --run_name wm-2b-4x3
python train_wordmaze_grpo.py --no_thinking --max_completion_length 256 --run_name wm-4b-direct   # the post's "direct-answer" regime
```

| flag | default | why |
|---|---|---|
| model | `Qwen/Qwen3.5-4B`, chat template, `enable_thinking=True` | only small model with a published non-trivial baseline. The template opens `<think>\n` for you; `--no_thinking` pre-closes it |
| LoRA | r=32, α=64, all linear layers | 4B full FT + 8 long completions won't fit 40 GB; `--full_ft` if you have 80 GB |
| `--lr` | 1e-5 | LoRA range 5e-6…2e-5. Full FT would be ~1e-6 |
| `--max_completion_length` | 1536 | thinking is long; watch `completions/clipped_ratio`. If > 0.4, go to 2048 and cut `per_device_train_batch_size` |
| batch | 4 × accum 4 = 16 completions = 2 prompts × 8 | small on purpose (long sequences); `steps` are cheap to add |
| `--mask_truncated` | off | DAPO-style: drop gradient from clipped completions. Turn on if clipped_ratio is high and training is noisy |
| dictionary | `top10k` (in `rewards.py`) | anti-junk-word |

**Time (unverified estimate).** 16 completions × ~1,500 tokens on a 4B with HF `generate` is the slow part:
plausibly 1.5–3 min per step on an A100 → 200 steps ≈ 5–10 h across sessions (use `--resume`). vLLM
(`--use_vllm`, vllm 0.27.1, README gotcha 3) is worth trying here more than anywhere else. Measure 5 steps first.

## 7. Step 3 — Read the curves

| key | expect | meaning |
|---|---|---|
| `rewards/format_reward/mean` | → 0.1 quickly | model always answers within budget |
| `rewards/ladder_reward/mean` | rises first | it learns "real word, one letter, right length, reach the goal" |
| `rewards/solve_reward/mean` | rises later, slower | the password: F/B bookkeeping while searching |
| `frac_reward_zero_std` | must fall below ~0.8 | otherwise no gradient — go back to Step 1 |
| `completions/clipped_ratio` | < 0.3 | thinking budget adequate |
| `completions/mean_length` | may rise | more explicit spelling-out in the thinking = good sign; look at samples |

Then the per-check eval after training on the **same bucket and split you measured before**:

```bash
python eval_wordmaze.py --model Qwen/Qwen3.5-4B --adapter runs/wordmaze-qwen3.5-4b-lora/final --tag 4b-grpo-4x3 --word_lengths 4 --move_counts 3
python eval_wordmaze.py --model Qwen/Qwen3.5-4B --adapter runs/wordmaze-qwen3.5-4b-lora/final --tag 4b-grpo-m34 --split test    # vs published 18.5 %
```

`results/*.json` gives every check's rate and `fully_valid_by_bucket`, so your table is the author's table with
a "+GRPO" column. Also check **generalisation**: train on 4×3, evaluate on 5×3 and 4×4 — did it learn the
*skill* (letter tracking) or memorise a small graph? 271 training puzzles is few; this question is the most
interesting one in the whole project.

## 8. What can go wrong (beyond the Countdown table)

| symptom | cause | fix |
|---|---|---|
| `ladder_reward` climbs, `solve_reward` flat | farming ladders, ignoring the password | reduce ladder to 0.15; add `--spaced`; check samples for whether it even writes F/B in its thinking |
| solve rate jumps on train but not on eval | memorising 271 puzzles | widen the bucket; eval on unseen buckets; lower LR |
| answers appear inside `<think>` only | model never closes the think block | it gets 0 by design; if `clipped_ratio` is high, raise the budget — it may be *trying* |
| junk words like `zzzz` in answers | you evaluated with `lenient` and it looks like cheating | that's why training used `top10k`; report both dictionaries |
| every completion identical | temperature too low / entropy collapsed | keep `--temperature 1.0` for training; sampling params in eval are separate |

## 9. Make it proof-of-work — and why this one is worth more than Countdown

The author literally ends the post with: "The dataset is on the Hub if you want to point a model at it. If you
beat 3.1-pro, I'd like to know how." His part two is unpublished. So the deliverable is the missing part two,
written by you:

1. **Baseline table** extended with 0.8B / 2B rows and the `--spaced` column (Step 1). Reproduce his 4B number on
   the m3-4 test split to show your harness matches his.
2. **The reward-hacking story with receipts.** Run one deliberately *naive* reward (`reaches_goal` + `within_max_moves`
   only) for 100 steps and show the two-word answers taking over; then your shaped reward on the same seed.
   Two curves, one figure. This is the single most shareable artefact in the project.
3. **Before → after per-check table** on the same bucket; **generalisation** to unseen buckets.
4. **Tokenization ablation under RL**: does `--spaced` training help more than `--spaced` prompting alone? Does a
   model trained *without* spaced prompts start spelling words out in its thinking on its own? (grep the
   completions for `c-o-l-d` / `c o l d` patterns over steps.)
5. **Interp hook:** the task is character-level. Probe whether the LoRA changed attention in the 6 full-attention
   layers vs the 18 DeltaNet layers (per-layer adapter norm), and whether per-letter tokens attract attention
   after training. Ties directly into your lesion-map work.

Post the thread, tag @immortaldip (he asked). Offer the PR: your `verifier.py` is a reusable grader for his
dataset repo, which currently ships only the generator.

---

### Sources (read 2026-09-07)
- Blog post: https://dipkumar.dev/posts/llm/wordmaze (June 3, 2026) — rules, verifier table, baselines, tokenization test, the hack, "part two" promise
- Dataset + generator: https://huggingface.co/datasets/immortal3/wordmaze (`create_dataset.py`, README; note the card says "exactly max_moves", the system prompt says "at most" — the verifier tracks both, the password length settles it)
- Blog index confirming no part two as of today: https://dipkumar.dev/blogs
- Qwen3.5-4B chat template: opens `<think>\n` by default; `enable_thinking=False` pre-closes it (checked with `transformers` 5.16.1)
- Unsloth Qwen3.5 fine-tuning guide (GRPO "works if you disable fast vLLM inference"): https://unsloth.ai/docs/models/qwen3.5/fine-tune
- TRL GRPO docs and defaults as in `../countdown/GUIDE.md`
- Referenced papers: RLVR / Tülu 3 (arXiv:2411.15124), GRPO (arXiv:2402.03300), DeepSeek-R1 (arXiv:2501.12948), "Limits of RLVR" (arXiv:2504.13837)
