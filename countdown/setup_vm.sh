#!/usr/bin/env bash
# One-shot, idempotent VM setup for the Countdown GRPO run on a Colab CLI session (run ON the VM via colab exec).
#   deps -> vllm (torch 2.13 swap, torchaudio/torchcodec removed) -> text-only Qwen3.5 checkpoint -> self-tests
# Log: /content/countdown/setup.log ; prints SETUP_OK / SETUP_FAILED at the end.
set -o pipefail
cd /content/countdown || exit 1
exec > setup.log 2>&1
set -x
python -c "import vllm" 2>/dev/null || {
  pip install -q -U "trl>=1.11" "transformers>=5.2" datasets peft accelerate wandb matplotlib flash-linear-attention huggingface_hub
  pip install -q "vllm==0.27.1"
  pip uninstall -y -q torchaudio torchcodec
}
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available()); from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla OK'); import vllm, trl, transformers; print('vllm', vllm.__version__, 'trl', trl.__version__, 'tf', transformers.__version__)" || { echo SETUP_FAILED; exit 1; }
[ -f /content/qwen3.5-0.8b-text/model.safetensors ] || python make_text_only_ckpt.py Qwen/Qwen3.5-0.8B-Base /content/qwen3.5-0.8b-text || { echo SETUP_FAILED; exit 1; }
python rewards.py | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo SETUP_OK
