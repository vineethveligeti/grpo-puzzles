#!/usr/bin/env bash
# After-training eval on the VM: base vs GRPO, greedy + pass@8, 1024 tokens (matches training), vLLM. Log: eval_all.log
cd /content/countdown; exec > eval_all.log 2>&1; set -x
GRPO=${GRPO:-aang2/qwen3.5-0.8b-countdown-grpo}; BASE=/content/qwen3.5-0.8b-text
python -u eval_countdown.py --vllm --model $GRPO --tag grpo400-greedy-1024 --max_new_tokens 1024
python -u eval_countdown.py --vllm --model $GRPO --tag grpo400-pass8-1024 --max_new_tokens 1024 --sample --k 8 --n 200
python -u eval_countdown.py --vllm --model $BASE --tag base-greedy-1024 --max_new_tokens 1024
python -u eval_countdown.py --vllm --model $BASE --tag base-pass8-1024 --max_new_tokens 1024 --sample --k 8 --n 200
python -u eval_countdown.py --vllm --model $GRPO --tag grpo400-greedy-512 --max_new_tokens 512
echo EVAL_ALL_DONE
