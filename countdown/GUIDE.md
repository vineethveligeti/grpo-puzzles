# Countdown × GRPO on Qwen3.5-0.8B-Base — step by step

> **TL;DR.** Take a *base* 0.8B model that has never seen an instruction, give it 3–4 numbers and a target,
> reward it only when its `<answer>` equation is correct, and run GRPO for a few hundred steps. You should watch
> (a) the format get learned in the first ~50 steps, (b) the solve rate climb from a few % toward tens of %,
> (c) the completions turn from rambling into "try 44+19=63, 63+35=98 ✓" style search. Total: one evening on a
> Colab A100, ~$5–15 of compute units. **Honest risk:** TinyZero found Qwen2.5-**0.5B** base *fails* to learn
> this; the tweet bets that the 2026 Qwen3.5-0.8B base is strong enough. Step 3 below tells you within 10 minutes
> whether that bet holds, and what to do if it doesn't.

Everything here was checked against primary sources on 2026-09-07 (links at the bottom). Code in this folder
runs end-to-end against `trl==1.12.0`, `transformers==5.16.1`.

---

## 0. The tweet, decoded

| Tweet line | What it actually means | Verified detail |
|---|---|---|
| grab **Qwen3.5-0.8B (base, not instruct)** | `Qwen/Qwen3.5-0.8B-Base`, Apache-2.0, released 2026-02-28 | 0.8B (0.75B text-only), 24 layers, hybrid 3×Gated-DeltaNet : 1×attention, vocab 248,320, 262k ctx. It is a *vision-language* checkpoint — see README gotcha 1 |
| grab **Jiayi Pan's Countdown-Tasks-3to4** | `Jiayi-Pan/Countdown-Tasks-3to4` | 490,364 rows; columns `target` (int 10–100) and `nums` (list of 3 or 4 ints in 1–99); 240,632 three-number and 249,732 four-number puzzles; every puzzle is solvable |
| **skip SFT entirely, go straight to RL** | R1-Zero style: no demonstrations, no chat template | Works because the reward is *executable* — a Python function decides correctness. (Author's reply in the thread: "SFT is basically instruct tuning... our outcome doesn't need it") |
| write a **reward fn: did it hit the target number** | rule-based: parse `<answer>…</answer>`, check each number used once, `eval`, compare | We add the TinyZero/Mini-R1 details the tweet skips: restricted characters, exact-once usage, float tolerance, `**` ban |
| run **GRPO for a few hundred steps** | TRL `GRPOTrainer`, ~300–500 optimizer steps | Reference curve (Mini-R1, 3B-Instruct, 8 gens): format learned by step ~50, 25% solved at 100, ~40% at 200, ~50% at 450 |
| watch it start **reasoning on its own** | completions grow verification/backtracking without ever being shown one | TinyZero: "the 3B base LM develops self-verification and search abilities all on its own" |

The recipe is **TinyZero** (Jiayi Pan et al., Jan 2025, veRL, Qwen2.5-3B, <$30) re-done with 2026 parts. Same
task, same prompt, same reward logic — smaller/newer model, TRL instead of veRL.

---

## 1. How the pieces fit

```mermaid
flowchart LR
    D[Countdown row<br/>nums=[44,19,35] target=98] --> P[TinyZero base prompt<br/>...Assistant: Let me solve this step by step.<br/>&lt;think&gt;]
    P --> G[Policy samples G=8 completions<br/>T=1.0, ≤512 tokens]
    G --> R1[format_reward<br/>0 / 0.5 / 1]
    G --> R2[answer_reward<br/>0 / 1]
    R1 & R2 --> A[Group-normalised advantage<br/>A_i = r_i − mean / std]
    A --> U[DAPO/GRPO update<br/>clip ε=0.2, β=0 no ref model]
    U -.every step.-> G
```

Per optimizer step with our defaults: `per_device_train_batch_size=8 × grad_accum=4` = **32 completions = 4 prompts × 8
generations**. 400 steps ≈ 1,600 prompts seen ≈ 12,800 completions scored.

---

## 2. Step 0 — Pre-flight (15 min)

1. Colab → Runtime → **A100** (40 GB). Check: `!nvidia-smi`.
2. Install (README "Shared setup"). Confirm the fast kernels are importable — transformers 5.16 gives **no warning**
   when they aren't: `python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla OK')"`.
   (`causal-conv1d` is optional; if its build fails, skip it — see README gotcha 2.)
3. `huggingface-cli login` (to push the result later) and `wandb login` (curves are the deliverable).
4. Get the code onto the VM: push this folder to a GitHub repo and `!git clone`, or upload the 4 `.py` files.
5. Run the self-tests: `python rewards.py` → `ALL PASS`; `python data.py` → prints an example prompt.

## 3. Step 1 — Look at the data (5 min, do not skip)

```python
from datasets import load_dataset
ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
ds[0]            # {'target': 98, 'nums': [44, 19, 35]}
```

Things worth knowing before you train: targets are small (10–100), 3-number puzzles are *much* easier than
4-number ones, division must come out exact (`(100/25)*(3+6)=36` is fine, `44*35/19` is not), and the dataset
has no solutions — you never need them, the verifier is the ground truth.

## 4. Step 2 — The reward function (already written; 10 min to understand)

`rewards.py` has two functions with TRL's signature `fn(prompts, completions, **dataset_columns)`; the dataset
columns `target` and `nums` arrive as keyword lists. TRL logs them separately as `rewards/format_reward/mean`
and `rewards/answer_reward/mean` — the second one **is your solve rate**.

| Completion | format | answer | why |
|---|---|---|---|
| `… </think>\n<answer> 44 + 19 + 35 </answer>` | 1.0 | 1.0 | the goal |
| same, then keeps rambling `User: …` | 0.5 | 1.0 | correct but didn't stop → teaches EOS |
| `<answer> 44 + 19 - 35 </answer>` | 1.0 | 0 | wrong value |
| `<answer> 44 + 44 + 10 </answer>` | 1.0 | 0 | numbers not used exactly once |
| `<answer> __import__('os') </answer>` | 1.0 | 0 | character filter → never reaches `eval` |
| `<answer> 9 ** 9 ** 9 </answer>` | 1.0 | 0 | `**` banned explicitly (passes the char filter, would hang) |
| no `</think>` | 0 | (checked anyway) | format must be `</think>` then one `<answer>` |

Three traps this design avoids, all seen in the wild: rewarding the *thinking* text (Mini-R1's format regex
only), letting the model use numbers twice (TinyZero's `validate_equation` catches this), and `eval` on
unrestricted strings.

## 5. Step 3 — Baseline eval: the 10-minute go/no-go (do this before any training)

```bash
python eval_countdown.py --model Qwen/Qwen3.5-0.8B-Base --tag base-greedy
python eval_countdown.py --model Qwen/Qwen3.5-0.8B-Base --tag base-pass8 --sample --k 8 --n 200
```

Read `results/base-pass8.json`:

| `pass@8` on the mix | meaning | do this |
|---|---|---|
| ≥ 5 % | plenty of signal: some groups of 8 will contain a success | train on the mix (`--curriculum none`) |
| 1–5 % | thin signal; most groups all-zero | `--curriculum 3only` for the first ~150 steps, then `none` |
| ≈ 0 % (also on `solve_rate_3nums`) | GRPO has nothing to amplify | switch to `Qwen/Qwen3.5-2B-Base`, **or** `Qwen/Qwen3.5-0.8B` (instruct; veRL's quickstart notes instruct models respond to RL signals better, and the 0.5B reproduction linked below only got traction from the instruct variant), **or** lower `temperature` to 0.7 for eval only to check it isn't a decoding artifact |

Why this matters: GRPO's advantage is $A_i = (r_i - \bar r)/\sigma_r$ computed *within* a group. If all 8
completions score 0, $\sigma_r = 0$ and the group contributes nothing. TRL reports the fraction of such groups
as `frac_reward_zero_std`; at step 0 you want it clearly below 1.0. This single number is also the honest
answer to "does 0.8B work?" — TinyZero's 0.5B failure is exactly the all-zero-groups regime.

## 6. Step 4 — Train

```bash
# A100 40 GB, full fine-tuning, ~400 steps
python train_countdown_grpo.py --report_to wandb --run_name cd-0.8b-base-mix

# thin signal? easy puzzles first
python train_countdown_grpo.py --curriculum 3only --max_steps 150 --run_name cd-0.8b-3only
python train_countdown_grpo.py --curriculum none  --max_steps 300 --resume runs/countdown-qwen3.5-0.8b-base/checkpoint-150

# L4 24 GB
python train_countdown_grpo.py --per_device_train_batch_size 4 --grad_accum 8      # same 32 completions/step
python train_countdown_grpo.py --lora --lr 2e-5                                     # or LoRA

# fast generation once the basics work (vllm==0.27.1; see README gotcha 3)
python train_countdown_grpo.py --use_vllm --vllm_gpu_memory_utilization 0.25
```

What the defaults are and why:

| flag | value | reason |
|---|---|---|
| `--lr` | 1e-6 | TRL default and TinyZero's actor LR; Mini-R1 used 5e-7; a 0.5B reproduction needed 5e-7 + grad-clip 0.5 to stay stable. Go *down* if `grad_norm` spikes, not up |
| `--num_generations` | 8 | Mini-R1's value; DeepSeekMath used 64. More = better advantage estimates, linearly more generation time |
| `--max_completion_length` | 512 | enough for 3–4 numbers; TinyZero used 1024 for 3B. Watch `completions/clipped_ratio` — if > 0.3 raise it |
| `--beta` | 0.0 | TRL default: no reference model (saves ~1.5 GB + a forward pass). Mini-R1 used 0.001, TinyZero used a KL term. Add `--beta 0.01` only if the model drifts into gibberish |
| `--temperature` | 1.0 | sampling diversity is what makes groups informative |
| loss | `dapo`, `scale_rewards="group"` | TRL 1.12 defaults |
| batch | 8 × 4 accum | 32 completions; the 248k-vocab logits are the memory hog, so this is the knob for OOM |

**Time.** Unverified estimate for HF `generate` on an A100 with `fla` installed: 30–60 s per step → 400 steps in
3–6 h. vLLM colocate typically cuts generation 3–5×. Measure your first 5 steps (`step_time` in the log) before
believing any of this.

**Memory (rough).** 0.75B bf16 weights 1.5 GB + fp32 AdamW states 6 GB + grads 1.5 GB ≈ 9 GB static; the
completion-token logits in fp32 for 8 sequences × 512 tokens × 248,320 vocab ≈ 4 GB plus softmax temporaries.
Fits a 40 GB A100 with headroom; 24 GB needs batch 4 or LoRA.

## 7. Step 5 — Reading the curves (the metric names are exactly what TRL prints)

| W&B / log key | expect | it means |
|---|---|---|
| `rewards/format_reward/mean` | 0.2 → 0.9+ within ~50 steps | model learned `</think>` then `<answer>`… then to stop |
| `rewards/answer_reward/mean` | few % → 20–50 % over 200–400 steps | **solve rate on training prompts** |
| `frac_reward_zero_std` | 0.8+ → falls | fraction of dead (all-same-reward) groups; must fall for learning to happen |
| `completions/mean_length` | often *dips* first (stops rambling) then *grows* (more verification) | the "aha" shape from the R1 paper |
| `completions/clipped_ratio` | < 0.2 | fraction hitting `max_completion_length`; high = raise the limit or the model is looping |
| `entropy` | slowly decreasing | collapsing to ~0 early = LR too high / diversity gone |
| `grad_norm` | smooth | spikes → halve LR |

Reference from Mini-R1 (Qwen2.5-**3B-Instruct**, 8 gens, 4×H100): format solved by step 50, 25 % at 100, ~40 %
at 200, 50 % at 450, and a *format shift* around step 200 from word-reasoning to "try combinations, check,
next" enumeration. Your 0.8B base will be slower and noisier; that's fine and worth writing about.

## 8. Step 6 — After: eval, samples, and what to look for

```bash
python eval_countdown.py --model runs/countdown-qwen3.5-0.8b-base/final --tag grpo400
python eval_countdown.py --model runs/countdown-qwen3.5-0.8b-base/final --tag grpo400-pass8 --sample --k 8 --n 200
```

Put `base-greedy` vs `grpo400` side by side: `solve_rate`, `solve_rate_3nums`, `solve_rate_4nums`,
`format_rate_clean`, `mean_completion_tokens`. Then read the 20 saved samples and grep the completions over
training (`log_completions=True` writes them) for behaviours nobody taught it: "let me verify", "that's not
right", "try instead", enumerating pairs, checking the running total. Count them per 50 steps — that count over
time *is* the "reasoning emerges" plot.

## 9. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| CUDA OOM at the first loss step | 248k-vocab logits | `--per_device_train_batch_size 4 --grad_accum 8`; then `--lora`; then `--max_completion_length 384` |
| Generation painfully slow (minutes per step) | `fla` not installed → silent torch fallback for 18 layers | `pip install flash-linear-attention`; verify with the import check in Step 0; restart runtime |
| `ValueError: There is no module or parameter named 'model' in Qwen3_5ForConditionalGeneration` | TRL #5269 (vLLM weight sync, text-only Qwen3.5) | `pip install vllm==0.27.1`; if it persists drop `--use_vllm` |
| `frac_reward_zero_std` stuck at 1.0 | no successes in any group | Step 3 decision table: curriculum `3only`, more generations (16), or a bigger/instruct model |
| format reward → 1 but answer reward flat at 0 for 100+ steps | model learned the *shape* but can't do arithmetic | same as above; also check `eval` of a few `<answer>` strings by hand — is the extraction right? |
| `completions/mean_length` pinned at 512 | looping / never emits `</answer>` | lower `temperature` to 0.8, or add `--beta 0.01`; check the completion isn't repeating |
| Loss is 0 and grad_norm 0 every step | all advantages zero (see above) or `num_generations` doesn't divide the batch | TRL requires `per_device_train_batch_size × grad_accum` divisible by `num_generations` |
| Model loads as `Qwen3_5ForConditionalGeneration` with a processor | you passed the model *name* to `GRPOTrainer` | keep the script's pattern: `AutoModelForCausalLM` + `AutoTokenizer` |

## 10. Make it proof-of-work

Minimum viable repo: this folder + `results/*.json` + one PNG of `answer_reward` vs step + a README with the
before/after table and 5 samples. Then the parts that make people stop scrolling:

- **The 0.8B question, answered with data.** TinyZero says 0.5B fails; the tweet says 0.8B works. Run both
  `Qwen3.5-0.8B-Base` and `Qwen3.5-2B-Base` with identical settings → one plot, two curves. That's a finding.
- **Base vs instruct** at 0.8B: `Qwen/Qwen3.5-0.8B` with the `qwen-instruct` template from TinyZero. Which learns faster? Does the base one reason "weirder"?
- **Format reward on/off**: drop `format_reward` — does the answer reward still climb? (R1-Zero used both.)
- **Group size** 8 vs 16 at fixed compute; **KL** 0 vs 0.01; **temperature** 1.0 vs 0.7.
- **Interp add-on that fits your `llm_lesion_map` work:** save checkpoints every 50 steps and plot the per-layer
  weight-delta norm $\|W_t - W_0\|_F$ — do the 6 attention layers or the 18 DeltaNet layers move more? Does the
  change concentrate late or early in the stack? Nobody has posted this for a hybrid-architecture model.
- **Token-level "aha" tracking:** frequency of `verify`, `wait`, `check`, `instead`, `=` in completions per step.

Write-up template: problem → recipe → the one plot → 3 samples (step 0 / 150 / 400) → ablation table → what
didn't work → cost. Post the thread, tag @TheGlobalMinima and @jiayi_pirate.

---

### Sources (all read 2026-09-07)
- Tweet: https://x.com/TheGlobalMinima/status/2096532361844609320 (2:34 AM · Sep 6, 2026)
- TinyZero repo + reward code: https://github.com/Jiayi-Pan/TinyZero — `verl/utils/reward_score/countdown.py`, `examples/data_preprocess/countdown.py` (prompt template used verbatim here). Deprecated in favour of veRL; "Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning."
- Mini-R1 (Phil Schmid, Jan 2025): https://www.philschmid.de/mini-deepseek-r1 and https://huggingface.co/blog/open-r1/mini-r1-contdown-game — reference curve and TRL reward-function pattern
- Model card + config: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- Dataset: https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
- TRL GRPO docs: https://huggingface.co/docs/trl/grpo_trainer ; defaults read from `trl==1.12.0` `GRPOConfig`
- transformers Qwen3.5 page (kernels note): https://huggingface.co/docs/transformers/model_doc/qwen3_5
- vLLM/TRL Qwen3.5 text-only bug: https://github.com/huggingface/trl/issues/5269 , https://github.com/vllm-project/vllm/issues/36275
- A 0.5B reproduction that needed LR 5e-7 / KL 0.005 / grad-clip 0.5: https://therawaideas.substack.com/p/steps-toward-reasoning-in-half-billion
- Someone who already GRPO'd Qwen3.5-0.8B on GSM8K (with an SFT warm-up, RTX 5090, 77 h): https://huggingface.co/zosmaai/Qwen3.5-0.8B-GRPO-Math
