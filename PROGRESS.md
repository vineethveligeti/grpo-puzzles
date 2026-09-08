# PROGRESS — grpo_puzzles

Source of truth for where this project stands. Update after every GPU run.

## Status (2026-09-08)

| | Countdown | Wordmaze |
|---|---|---|
| code | done, CPU smoke-tested | done, CPU smoke-tested |
| baseline eval on GPU | **not run** ← next | **not run** |
| training run | not started | not started |
| write-up / thread | — | — |

Environment on Colab (2026-09-08): `trl`, `transformers`, `datasets`, `peft`, `accelerate`, `wandb`,
`flash-linear-attention`, `wordfreq` installed fine. `causal-conv1d` failed to build (from-source CUDA
extension) — **skipped on purpose**, see README gotcha 2; it only accelerates a width-4 depthwise conv and
falls back independently to `F.conv1d`.

## What was verified (2026-09-07, sandbox: trl 1.12.0, transformers 5.16.1, datasets 5.0.1)

- `countdown/rewards.py` — 11 self-test cases pass (correct, junk-after-answer, wrong value, number reused,
  number missing, no `</think>`, code injection, `**`, inexact division, zero-division, 4-number exact division).
- `wordmaze/verifier.py` — 13 cases pass incl. the blog's two-word hack, `shark -> shore` two-letter step,
  spaced letters, alternate valid solution, unfinished `<think>`; `top10k` dictionary rejects `zzzz`, `lenient` accepts it.
- `wordmaze/rewards.py` — totals 1.4 / 0.4 / 0.1 / 0.0 for solve / ladder-only / hack / nothing.
- Both `train_*.py` ran 2 optimizer steps end-to-end on CPU with a tiny model (`--cpu_smoke`); TRL logged
  `rewards/<fn>/mean`, `frac_reward_zero_std`, `completions/clipped_ratio`, saved a checkpoint.
- Both `eval_*.py` ran on CPU and wrote `results/<tag>.json`.
- Both notebooks validate with `nbformat`.
- Qwen3.5-0.8B-Base tokenizer: `<think>`/`</think>` single non-special tokens; EOS `<|endoftext|>` (248044);
  TinyZero base prompt = 141 tokens. Qwen3.5-4B chat template opens `<think>\n` unless `enable_thinking=False`.
- `AutoModelForCausalLM` on a `qwen3_5` config → `Qwen3_5ForCausalLM` (text-only). TRL's `create_model_from_path`
  would instead follow `architectures[0]` → full VLM + processor + fp32.
- Dataset facts: Countdown 490,364 rows (240,632 × 3 nums, 249,732 × 4 nums), targets 10–100.
  Wordmaze m3-4 train buckets (len, moves): (4,3) 271 · (4,4) 258 · (5,3) 268 · (5,4) 272 · (6,3) 264 · (6,4) 267.

## Corrections log

- 2026-09-08: removed the instruction to watch for a *"fast path is not available"* log line — that warning does
  not exist in transformers 5.16 (it falls back silently). Replaced with an explicit import check:
  `python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla OK')"`.

## Next actions (in order)

1. Colab A100: `cd countdown && python eval_countdown.py --model Qwen/Qwen3.5-0.8B-Base --tag base-pass8 --sample --k 8 --n 200`
   → read `pass@8`, pick the branch in `countdown/GUIDE.md` §5.
2. Train (`train_countdown_grpo.py --report_to wandb`), check first 5 steps: `step_time`, `frac_reward_zero_std` < 1.
3. After-eval, `curves.png`, 5 samples → README table → thread.
4. Wordmaze baseline table (0.8B / 2B / 4B × plain / spaced) on the 4×3 bucket.
5. Naive-vs-shaped reward "hacking receipt" run, then real training.

## Decisions

- Countdown: full fine-tune of 0.8B base, `beta=0`, DAPO loss, 4 prompts × 8 gens per step, `use_vllm=False` default.
- Wordmaze: 4B instruct + LoRA r=32, thinking on, train on `top10k` dictionary, eval on `lenient` (comparable to published numbers).
- Time estimates in the guides are unverified until the first 5 GPU steps are measured.
