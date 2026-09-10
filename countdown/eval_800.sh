#!/usr/bin/env bash
cd /content/countdown; exec > eval_800.log 2>&1; set -x
GRPO=aang2/qwen3.5-0.8b-countdown-grpo
python -u eval_countdown.py --vllm --model $GRPO --tag grpo800-greedy-1024 --max_new_tokens 1024
python -u eval_countdown.py --vllm --model $GRPO --tag grpo800-pass8-1024 --max_new_tokens 1024 --sample --k 8 --n 200
python -u eval_countdown.py --vllm --model $GRPO --tag grpo800-greedy-512 --max_new_tokens 512
echo EVAL_ALL_DONE
