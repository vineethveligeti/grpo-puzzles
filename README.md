# GRPO Puzzles — two "RL project to showcase" recipes, verified and ready to run

Two small reinforcement-learning-from-verifiable-rewards (RLVR) projects, each a full pipeline
(data → prompt → reward function → GRPO → before/after eval), written against **TRL 1.12 /
transformers 5.16 (Sep 2026)** and smoke-tested end-to-end in a sandbox (CPU, tiny model).
GPU runs are yours to do on Colab Pro.

| | `countdown/` | `wordmaze/` |
|---|---|---|
| Origin | [@TheGlobalMinima's tweet](https://x.com/TheGlobalMinima/status/2096532361844609320) — the TinyZero recipe | [@immortaldip's reply](https://dipkumar.dev/posts/llm/wordmaze) — "try this instead" |
| Task | Use numbers `[44, 19, 35]` once each with `+ - * /` to hit `98` | Word ladder `cold → warm` in exactly N one-letter moves whose F/B letter-direction pattern spells a password |
| Data | `Jiayi-Pan/Countdown-Tasks-3to4` — 490k puzzles | `immortal3/wordmaze` — 2 × 2,000 puzzles (`m3-4` easy, `m4-6` hard) |
| Model | `Qwen/Qwen3.5-0.8B-Base` (base, no SFT) | `Qwen/Qwen3.5-4B` instruct, thinking on, LoRA (0.8B is ~0% here) |
| Difficulty for the model | Easy-ish → you *will* see reward go up | Hard → published 4B baseline is 18.5% (easy config), 2.5% (hard) |
| Reward | correct equation 0/1 (+ format) | 10 boolean checks; shaped 0.1 / 0.3 / 1.0 designed against the known hack |
| Prior art | TinyZero (Jan 2025), Mini-R1 (Phil Schmid) | Env + baselines only. **Nobody has published the GRPO training story yet** |
| Time to first curve | ~1 evening on an A100 | ~1–2 weeks, needs the baseline-first discipline |
| Portfolio angle | "I reproduced the aha moment on a 0.8B and measured what changed" | "I ran GRPO on an env whose author showed naive RL makes the model worse — here is a reward that doesn't" |

**Do Countdown first.** It is the guaranteed win and it debugs your whole stack (Colab, TRL, Qwen3.5
kernels, W&B). Then Wordmaze, where the interesting science is.

## Files

```
grpo_puzzles/
├── README.md                 <- you are here (shared setup + gotchas)
├── GUIDE.html                <- visual companion: GRPO loop, reward trees, expected curves, decision flow
├── requirements.txt
├── countdown/
│   ├── GUIDE.md              <- step-by-step
│   ├── data.py               <- dataset + TinyZero base prompt
│   ├── rewards.py            <- format_reward + answer_reward   (python rewards.py -> self-test)
│   ├── train_countdown_grpo.py
│   ├── eval_countdown.py     <- solve rate on 500 held-out puzzles, before/after
│   └── countdown_grpo_colab.ipynb
└── wordmaze/
    ├── GUIDE.md
    ├── data.py               <- dataset + chat/base prompts + spelled-out-letters ablation
    ├── verifier.py           <- the 10 checks, 2 dictionaries   (python verifier.py -> self-test)
    ├── rewards.py            <- format 0.1 + valid-ladder 0.3 + solve 1.0
    ├── train_wordmaze_grpo.py
    ├── eval_wordmaze.py      <- per-check table like the blog post's, per (length, moves) bucket
    └── wordmaze_grpo_colab.ipynb
```

## Shared setup (Colab Pro)

```bash
pip install -U "trl>=1.11" "transformers>=5.2" datasets peft accelerate wandb
pip install flash-linear-attention        # Qwen3.5 Gated-DeltaNet fast kernels (Triton) — see gotcha 2
# pip install causal-conv1d               # optional & minor; often fails to build on Colab — skip, or try --no-build-isolation
pip install wordfreq                      # wordmaze only
# optional fast generation (see gotcha 3):
pip install "vllm==0.27.1"
```

GPU: **A100 40 GB** is the comfortable choice for both projects. L4 (24 GB) works for Countdown with
`--per_device_train_batch_size 4 --grad_accum 8` or `--lora`. T4 has no bf16 — avoid.

## Five Qwen3.5 gotchas the tweet doesn't mention (all verified Sep 7, 2026)

1. **It's a vision-language checkpoint.** `Qwen/Qwen3.5-0.8B-Base` has `architectures: ["Qwen3_5ForConditionalGeneration"]`
   and a vision tower. Load it with `AutoModelForCausalLM` → transformers resolves `qwen3_5` to
   `Qwen3_5ForCausalLM` (text-only, vision weights skipped). If you instead pass the model *name* string to
   `GRPOTrainer`, TRL follows `config.architectures[0]`, loads the full VLM + `AutoProcessor`, treats it as a
   VLM, and (per TRL docs) defaults the dtype to **float32**. Our scripts instantiate the model themselves in bf16.
2. **Hybrid linear attention needs kernels.** 18 of 24 layers are Gated DeltaNet. Without the `fla` package
   transformers falls back to a slow, memory-hungry PyTorch path — silently in 5.16; **5.17 does log**
   `chunk_gated_delta_rule is falling back to its reference PyTorch implementation` (seen 2026-09-09). Check explicitly anyway: `python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla OK')"`.
   `pip install flash-linear-attention` covers the heavy op. `causal-conv1d` is a *second, minor* kernel (a depthwise
   conv of width 4); each kernel falls back independently, so if its from-source build fails you lose very little —
   skip it, or get a prebuilt one via `pip install kernels` + `from_pretrained(..., use_kernels=True)` (Hub repo
   `kernels-community/mamba-ssm`).
3. **TRL + vLLM + Qwen3.5 text-only is still broken in vllm 0.27.1 — but there is a 2-minute workaround.**
   ([TRL #5269](https://github.com/huggingface/trl/issues/5269)) vLLM instantiates the VLM class from the Hub
   checkpoint, TRL's colocated weight sync sends text-only names → *"no module or parameter named 'model'"*.
   vLLM 0.27.1 *does* have a text-only `Qwen3_5ForCausalLM`; it just needs a checkpoint that says so:
   `python countdown/make_text_only_ckpt.py Qwen/Qwen3.5-0.8B-Base /content/qwen3.5-0.8b-text`, then pass that dir as
   `--model` together with `--use_vllm`. Verified 2026-09-08 on an A100: raw vLLM decode **15.4k tok/s** vs **~1.6k tok/s**
   for HF `generate` (512 sequences in flight, 1024 tokens). Install notes: `vllm==0.27.1` pins `torch==2.13.0` (Colab ships
   2.11) — the swap works with `flash-linear-attention` and transformers 5.16.1, but you must
   `pip uninstall -y torchaudio torchcodec` afterwards (they stay on the CUDA-12.8 build and break `import torch`-dependent
   imports). Use `--vllm_gpu_memory_utilization 0.45`; 0.3 leaves no room for vLLM's KV/state cache next to the trainer.
4. **248k-token vocabulary.** The logits tensor (batch × completion_len × 248,320) dominates memory at the
   loss step, not the 0.8B weights. That's why `per_device_train_batch_size` matters more than you'd expect;
   halve it before reaching for LoRA.
5. **`<think>` / `</think>` are single tokens but not "special"** → they survive TRL's `skip_special_tokens=True`
   decode, so format rewards can see them. The base model's EOS/pad is `<|endoftext|>` (id 248044); the instruct
   model's EOS is `<|im_end|>`.

## What "GRPO" is doing (30-second version — full visual in GUIDE.html, math in ../rl_algorithms_tour.html)

For each prompt, sample a **group** of G=8 completions. Score each with the rule-based reward. Turn scores into
advantages by normalising *within the group*: $A_i = (r_i - \bar r)/\sigma_r$ (TRL `scale_rewards="group"`). Push
up the log-probability of tokens in above-average completions, push down the rest, with a PPO-style clip
($\epsilon=0.2$) and — in TRL 1.12's default `loss_type="dapo"` — token-level averaging across the group.
No critic, no reward model, and with `beta=0.0` no reference model either. **Consequence to remember:** a group
whose 8 completions all score the same contributes *zero* gradient. TRL logs this as `frac_reward_zero_std`
— when it sits at 1.0 you have no signal, and the fix is a curriculum or a stronger base model, not more steps.

## Proof-of-work checklist (what to publish)

- A GitHub repo per project (or this folder as one repo) with README: before → after table, reward/solve-rate curve,
  5 qualitative samples, exact command lines, cost.
- W&B project public link (`--report_to wandb`).
- Trained adapter/model on the Hub with a model card that states the eval protocol.
- One X thread each (@findingsignal12): the curve, one "aha" sample, one honest failure. Tag the original posters —
  both tweets have engaged audiences and the Wordmaze author explicitly asked to hear from people who train on it.
- Ablations are where the credibility comes from — each GUIDE lists 4–6 cheap ones.
