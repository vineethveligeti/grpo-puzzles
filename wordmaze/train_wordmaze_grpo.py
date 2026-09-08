#!/usr/bin/env python
"""GRPO on Wordmaze (word ladder + F/B password) with a small Qwen3.5 instruct model, LoRA, thinking on.

    # 1) always measure the baseline first (eval_wordmaze.py) — GRPO can only amplify a non-zero solve rate
    python train_wordmaze_grpo.py                                  # Qwen3.5-4B, LoRA, m3-4, 4-letter/3-move bucket
    python train_wordmaze_grpo.py --model Qwen/Qwen3.5-2B          # cheaper, lower baseline
    python train_wordmaze_grpo.py --word_lengths 4 5 --move_counts 3 4   # widen the curriculum
    python train_wordmaze_grpo.py --spaced                         # spelled-out letters ablation ("c o l d")
    python train_wordmaze_grpo.py --cpu_smoke                      # wiring check, CPU, 2 steps

Verified against: trl==1.12.0, transformers==5.16.1 (Sep 2026).

Why these defaults (details in GUIDE.md):
  * 4B instruct in THINKING mode: the only small open model with a published non-trivial baseline
    (18.5% fully_valid on m3-4, 2.5% on m4-6 — dipkumar.dev).  0.8B/2B may sit at ~0% -> no signal.
  * LoRA r=32 on all linear layers: 4B full fine-tuning + 8 long completions does not fit a 40 GB card.
  * Rewards (rewards.py): format 0.1 + valid-ladder 0.3 + full-solve 1.0, strict top-10k dictionary
    during training.  This is what makes the two-word shortcut hack worthless.
  * Completions are long (thinking).  max_completion_length=1536 is a compromise; measure the
    truncation rate in the logs (completions/clipped_ratio) and raise it if it is high.
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import rewards
from data import load_wordmaze


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--style", default="chat", choices=["chat", "base"], help="chat = instruct model + chat template")
    p.add_argument("--no_thinking", action="store_true", help="enable_thinking=False (direct answer mode)")
    p.add_argument("--config", default="m3-4", choices=["m3-4", "m4-6"])
    p.add_argument("--word_lengths", type=int, nargs="*", default=[4])
    p.add_argument("--move_counts", type=int, nargs="*", default=[3])
    p.add_argument("--spaced", action="store_true", help="spell words letter-by-letter in the prompt")
    p.add_argument("--output_dir", default="runs/wordmaze-qwen3.5-4b-lora")
    p.add_argument("--run_name", default=None)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5, help="LoRA: 5e-6..2e-5; full FT: ~1e-6")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--full_ft", action="store_true", help="disable LoRA (needs a big GPU)")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--mask_truncated", action="store_true", help="DAPO-style: no gradient from truncated completions")
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--report_to", default="none")
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None)
    p.add_argument("--cpu_smoke", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    on_cpu = a.cpu_smoke or not torch.cuda.is_available()
    if a.cpu_smoke:
        a.model = "HuggingFaceTB/SmolLM2-135M-Instruct"
        a.max_steps, a.num_generations, a.per_device_train_batch_size, a.grad_accum = 2, 2, 2, 1
        a.max_completion_length, a.save_steps = 24, 1000
        a.output_dir = os.path.join(a.output_dir, "cpu_smoke")

    thinking = not a.no_thinking
    # Our prompts open a <think> block when: base template (always) or chat template with thinking on.
    rewards.set_think_open(a.style == "base" or thinking)

    train_ds = load_wordmaze(a.config, "train", style=a.style, use_spaced=a.spaced,
                             word_lengths=a.word_lengths or None, move_counts=a.move_counts or None, seed=a.seed)
    print(f"train puzzles: {len(train_ds)}  (config={a.config}, lengths={a.word_lengths}, moves={a.move_counts})")

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32 if on_cpu else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)   # qwen3_5 -> Qwen3_5ForCausalLM (text-only)
    print(f"loaded {type(model).__name__}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    peft_config = None
    if not a.full_ft:
        from peft import LoraConfig
        peft_config = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    cfg = GRPOConfig(
        output_dir=a.output_dir, run_name=a.run_name, seed=a.seed,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        max_grad_norm=1.0,
        bf16=not on_cpu, use_cpu=on_cpu, gradient_checkpointing=not on_cpu,
        # GRPO
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_length,
        temperature=a.temperature,
        beta=a.beta,
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=a.mask_truncated,
        chat_template_kwargs={"enable_thinking": thinking} if a.style == "chat" else None,
        # generation backend
        use_vllm=a.use_vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization,
        # logging
        logging_steps=1, log_completions=True, num_completions_to_print=2,
        save_steps=a.save_steps, save_total_limit=3, report_to=a.report_to,
    )

    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=rewards.REWARD_FUNCS,
                          args=cfg, train_dataset=train_ds, peft_config=peft_config)
    trainer.train(resume_from_checkpoint=a.resume)

    final_dir = os.path.join(a.output_dir, "final")
    trainer.save_model(final_dir)          # LoRA adapter (or full weights with --full_ft)
    tok.save_pretrained(final_dir)
    print(f"saved to {final_dir}")


if __name__ == "__main__":
    main()
