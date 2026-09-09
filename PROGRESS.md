# PROGRESS — grpo_puzzles

Source of truth for where this project stands. Update after every GPU run.

## Status (2026-09-08)

| | Countdown | Wordmaze |
|---|---|---|
| code | done, CPU smoke-tested | done, CPU smoke-tested |
| baseline eval on GPU | **done 2026-09-08** (see below) | **not run** |
| training run | **running** since 2026-09-08 ~21:30 PT (2nd attempt, vLLM): `cd-0.8b-base-3only-g16-vllm`, see below. 1st attempt (HF generate, 8 gens) stopped at step 32 | not started |
| write-up / thread | — | — |

Environment on Colab (2026-09-08): `trl`, `transformers`, `datasets`, `peft`, `accelerate`, `wandb`,
`flash-linear-attention`, `wordfreq` installed fine. `causal-conv1d` failed to build (from-source CUDA
extension) — **skipped on purpose**, see README gotcha 2; it only accelerates a width-4 depthwise conv and
falls back independently to `F.conv1d`.

## Countdown baseline (A100-40GB, 2026-09-08) — Qwen3.5-0.8B-Base, TinyZero prompt

| eval | solve | 3nums | 4nums | pass@8 | clean format | any format | mean tokens | time |
|---|---|---|---|---|---|---|---|---|
| greedy, 500 puzzles, 512 tok | 0.000 | 0.000 | 0.000 | — | 0.006 | 0.008 | 511.5 | 434 s |
| sample T=1, 8×200 puzzles, 512 tok | 0.000 (first sample) | 0.000 | — | **0.025** | 0.003 | 0.005 | 501.3 | 1298 s |

Reading: the base model *thinks like a long-CoT model* ("We are given numbers… Let's think…") and almost never
closes `</think>`/`<answer>` within 512 tokens — 98% of completions are cut off. pass@8 = 2.5% → the guide's
**"thin signal" branch → `--curriculum 3only`**.

5-step timing run (3only, `--max_completion_length 1024`, batch 4 × accum 8; batch 8 × 1024 **OOMs** in the fla
Gated-DeltaNet backward on 40 GB):

| step | clipped_ratio | format mean | answer mean | frac_reward_zero_std | step_time |
|---|---|---|---|---|---|
| 1 | 0.91 | 0.031 | 0 | 0.75 | 95 s (Triton autotune) |
| 2 | 0.88 | 0.016 | 0 | 0.75 | 51 s |
| 3 | 0.91 | 0 | 0 | 1.00 | 50 s |
| 4 | 0.75 | 0 | **0.063** | 0.75 | 51 s |
| 5 | 0.84 | 0.031 | 0 | 0.75 | 51 s |

So: ~50 s/step → 400 steps ≈ 5.6 h. Even at 1024 tokens ~85–90% of completions are clipped (terminated ones average
~680 tokens), and roughly 1 completion in 32 either formats correctly or solves — enough that most groups have a
non-zero-std reward, i.e. GRPO has *something* to push on. The format reward is what has to teach it to stop.

GPU-utilisation work (2026-09-08 evening, via the Colab CLI session `cd-a100`):

| generation config | first opt step (incl. generation) | later opt steps | per 32 completions |
|---|---|---|---|
| HF generate, 32 per call, 32/step (browser run, 32 steps done then interrupted) | 52 s | — | 52 s |
| HF generate, 512 per call, 16 gens/prompt, 128/step | 313 s | 15 s ×3 | 22 s |
| **vLLM colocated (sleep mode, util 0.35), 512 per call, 16 gens, 128/step** | 201 s (incl. vLLM start) / **78 s** steady | 15 s ×3 | **7.7 s** |

Raw vLLM decode on the text-only checkpoint: 512 completions, 487k tokens in 31.7 s = **15.4k tok/s** (HF generate: ~1.6k).
Fixes needed to get there (all in the repo now): `make_text_only_ckpt.py` (TRL #5269 workaround), `pip uninstall torchaudio torchcodec`
after the vllm install, `vllm_enable_sleep_mode=True` (0.45 util without sleep OOMed in the step-2 backward), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Colab reclaims CLI-created A100 VMs after exactly 60 min** (seen twice, 2026-09-08; keep-alive daemon healthy, 75 compute
units left; the browser-created runtime lived >2 h). Run #2 died at step 53 with its checkpoint. Mitigation, now in place:
`--hub_repo aang2/qwen3.5-0.8b-countdown-grpo` (private; every 50-step checkpoint pushed to `last-checkpoint/`) +
`--resume auto` + `supervise.sh` on the Mac (recreates the session, re-runs `setup_vm.sh`, pushes tokens from `~/.zshrc`,
relaunches; W&B run id `cd08b-3only-g16-vllm` with `WANDB_RESUME=allow` so the curve continues across VMs).
Run #3 started 2026-09-08 22:20 PT under `caffeinate`; log: `countdown/supervise.log`.

**Main run #2 launched 2026-09-08 ~21:30 PT on CLI session `cd-a100`** (died at step 53, see above) (`train_run.sh`): 3only, len 1024, 16 gens/prompt, 512 per
generation, 128 per optimizer step, 400 steps, vLLM, W&B *offline* (`cd-0.8b-base-3only-g16-vllm`; sync with `wandb sync` once
`WANDB_API_KEY` is set in the kernel). Expected ~3.5 h. Log: `/content/countdown/train.log`; checkpoints every 50 steps.

HF `generate` on this model runs ~1.6k tok/s even at 512 sequences in flight (training fwd+bwd runs ~8k tok/s), so the
A100 sits at ~20% power. Next lever: vLLM colocated generation (`--use_vllm`, vllm==0.27.1 pins torch 2.13 vs Colab's 2.11 —
tested on the CLI session before committing to it). With 16 gens/prompt, 34% of groups had non-uniform reward at step 1
(vs 25–50% with 8).

The browser runtime was deleted after the baseline JSONs were saved to `countdown/results/` (committed).
The interrupted browser run (32 steps, 3only/len1024/8 gens) is still on W&B as `cd-0.8b-base-3only-len1024`.

Notebook (self-contained, uploaded to Drive): https://colab.research.google.com/drive/1dPoHlXynzgw2vgulGhAgavdAklal27Ef
Colab Secrets `HF_TOKEN` + `WANDB_API_KEY` are set; logins verified in-notebook.
W&B run: https://wandb.ai/vineethveligeti-asu/grpo-countdown/runs/s9bynohq

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

1. ~~Baseline pass@8~~ done: 2.5% → 3only.  ~~Timing run~~ done: 50 s/step at 1024 tokens.
2. **Watch the running 400-step job** (W&B `grpo-countdown`): `rewards/format_reward/mean` should climb first, then
   `completions/mean_length` should *drop* (learning to stop), then `rewards/answer_reward/mean`. If format is still
   ~0 after ~100 steps, stop and try `Qwen/Qwen3.5-0.8B` (instruct) or 2B-Base per GUIDE §5.
   Checkpoints every 50 steps in `runs/countdown-qwen3.5-0.8b-base/`; keep the Colab tab open (Pro has no background execution).
3. After-eval (same 500 puzzles, also at `--max_new_tokens 1024` to match training), `curves.png`, 5 samples → README table → thread.
4. Move future runs to the **Colab CLI** (`~/.local/bin/colab`, installed 2026-09-08): `colab new --gpu A100`,
   `colab install -r requirements.txt`, `colab exec -f train_countdown_grpo.py`, `colab download`, `colab log`.
   Streams stdout, has a keep-alive daemon, no browser needed. One-time OAuth copy-paste login is required first.
4. Wordmaze baseline table (0.8B / 2B / 4B × plain / spaced) on the 4×3 bucket.
5. Naive-vs-shaped reward "hacking receipt" run, then real training.

## Decisions

- Countdown: full fine-tune of 0.8B base, `beta=0`, DAPO loss, 4 prompts × 8 gens per step, `use_vllm=False` default.
- Wordmaze: 4B instruct + LoRA r=32, thinking on, train on `top10k` dictionary, eval on `lenient` (comparable to published numbers).
- Time estimates in the guides are unverified until the first 5 GPU steps are measured.
