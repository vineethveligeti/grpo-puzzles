#!/usr/bin/env python
"""GRPO on the Countdown game with a *base* (not instruct) Qwen3.5 model.  No SFT.

    python train_countdown_grpo.py                       # A100-40GB defaults
    python train_countdown_grpo.py --curriculum 3only    # if the base model solves ~0% at step 0
    python train_countdown_grpo.py --lora --per_device_train_batch_size 4 --grad_accum 8   # L4 / 24 GB
    python train_countdown_grpo.py --cpu_smoke           # 2 tiny steps on CPU, wiring check only

Verified against: trl==1.12.0, transformers==5.16.1, datasets==5.0.1 (Sep 2026).

Key facts baked in (see GUIDE.md for the why):
  * We instantiate the model ourselves with AutoModelForCausalLM -> for a `qwen3_5` config this
    resolves to Qwen3_5ForCausalLM (text-only; the vision tower is simply not loaded).  If you pass
    the model *name* to GRPOTrainer instead, TRL follows config.architectures[0] and loads the full
    vision-language class + AutoProcessor, and (per TRL docs) defaults the dtype to float32.
  * Prompt = TinyZero base template ending in "<think>", plain string (TRL "standard" format).
  * Rewards: format_reward (0/0.5/1) + answer_reward (0/1). Solve rate = rewards/answer_reward/mean.
  * beta=0.0 (TRL default): no reference model, saves memory; loss_type defaults to "dapo".
  * Effective batch = per_device_train_batch_size * grad_accum completions
                    = 8 * 4 = 32 completions = 4 prompts x 8 generations per optimizer step.
  * GPU utilisation: a GRPO step is ~80% autoregressive generation, and HF generate on a 0.8B model is
    kernel-launch-bound, so the A100 sits at ~20% power with 32 sequences in flight. --generation_batch_size 128
    decodes 128 sequences per call (spread over 4 optimizer steps, standard "generation batch > train batch",
    PPO clip handles the mild off-policy-ness) for ~the same wall-clock as 32. --use_vllm goes further.
"""

from __future__ import annotations

import argparse
import os

# Reduce allocator fragmentation (a 40 GB A100 lost 6 GB to "reserved but unallocated" in the vLLM-colocated run).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data import load_countdown
from rewards import answer_reward, format_reward


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--output_dir", default="runs/countdown-qwen3.5-0.8b-base")
    p.add_argument("--run_name", default=None)
    p.add_argument("--n_train", type=int, default=20_000, help="prompts to draw from (490k available)")
    p.add_argument("--curriculum", default="none", choices=["none", "3only", "4only"])
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-6, help="full FT: 5e-7..3e-6; LoRA: 1e-5..5e-5")
    p.add_argument("--beta", type=float, default=0.0, help="KL coef; 0 = no reference model")
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--generation_batch_size", type=int, default=None,
                   help="completions generated per HF-generate call, spread over several optimizer steps "
                        "(TRL steps_per_generation). Default = 1 optimizer step's worth "
                        "(per_device_train_batch_size * grad_accum). Decoding a 0.8B model is launch-bound, "
                        "so 128 costs barely more wall-clock than 32 -> ~3-4x faster steps. Must be a "
                        "multiple of per_device_train_batch_size and of num_generations.")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="disable gradient checkpointing (saves the recompute forward; fits on 40 GB at batch 4)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora", action="store_true", help="LoRA r=32 on all linear layers instead of full FT")
    p.add_argument("--use_vllm", action="store_true", help="colocated vLLM generation (needs vllm>=0.27, see GUIDE)")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    p.add_argument("--no_vllm_sleep", action="store_true",
                   help="keep vLLM resident during the train phase (default: sleep mode frees its KV cache + weights between generations so the trainer gets the whole GPU)")
    p.add_argument("--report_to", default="none", help="'wandb' or 'none'")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="path to a checkpoint dir to resume from")
    p.add_argument("--cpu_smoke", action="store_true", help="tiny model, 2 steps, CPU: checks the wiring only")
    return p.parse_args()


def main():
    a = parse_args()
    on_cpu = a.cpu_smoke or not torch.cuda.is_available()

    if a.cpu_smoke:  # wiring check: tiny real model, tiny batch, 2 optimizer steps
        a.model = "HuggingFaceTB/SmolLM2-135M"
        a.max_steps, a.num_generations, a.per_device_train_batch_size, a.grad_accum = 2, 2, 2, 1
        a.max_completion_length, a.n_train, a.save_steps = 24, 8, 1000
        a.output_dir = os.path.join(a.output_dir, "cpu_smoke")

    train_ds, _eval_ds = load_countdown(n_train=a.n_train, curriculum=a.curriculum, seed=a.seed)
    print(f"train prompts: {len(train_ds)}  | example:\n{train_ds[0]['prompt']}\n")

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"                     # GRPO pads prompts on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float32 if on_cpu else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    print(f"loaded {type(model).__name__} with {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    peft_config = None
    if a.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    cfg = GRPOConfig(
        output_dir=a.output_dir,
        run_name=a.run_name,
        seed=a.seed,
        # --- optimisation ---
        learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        max_grad_norm=1.0,
        bf16=not on_cpu,
        use_cpu=on_cpu,
        gradient_checkpointing=(not on_cpu) and (not a.no_grad_ckpt),
        # --- GRPO ---
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_length,
        generation_batch_size=a.generation_batch_size,
        temperature=a.temperature,
        beta=a.beta,
        loss_type="dapo",
        scale_rewards="group",
        # --- generation backend ---
        use_vllm=a.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=a.use_vllm and not a.no_vllm_sleep,
        # --- logging / saving ---
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        save_steps=a.save_steps,
        save_total_limit=3,
        report_to=a.report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[format_reward, answer_reward],
        args=cfg,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=a.resume)

    final_dir = os.path.join(a.output_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"saved to {final_dir}")


if __name__ == "__main__":
    main()
