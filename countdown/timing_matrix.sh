#!/usr/bin/env bash
# 8-step timing runs from the text-only base ckpt. Log per config: timing_<name>.log ; summary: timing_matrix.log
cd /content/countdown; exec > timing_matrix.log 2>&1
COMMON="--model /content/qwen3.5-0.8b-text --max_steps 8 --save_steps 1000 --report_to none --curriculum 3only --max_completion_length 1024 --num_generations 16 --use_vllm --vllm_gpu_memory_utilization 0.35"
run() { name=$1; shift; echo "=== $name: $*"; rm -rf runs/tm_$name; python -u train_countdown_grpo.py $COMMON --output_dir runs/tm_$name "$@" > timing_$name.log 2>&1; python - "$name" <<'PY'
import re,sys
name=sys.argv[1]; ls=[l for l in open(f'timing_{name}.log') if "'step_time'" in l]
t=[float(re.search(r"'step_time': '([^']+)'",l).group(1)) for l in ls]
err=[l.strip()[:160] for l in open(f'timing_{name}.log') if 'Error' in l or 'Traceback' in l][-1:]
print(f"RESULT {name}: steps={len(t)} times={[round(x) for x in t]} err={err}")
PY
}
run B_nockpt        --per_device_train_batch_size 4 --grad_accum 32 --generation_batch_size 512 --no_grad_ckpt
run C_nockpt_bs8    --per_device_train_batch_size 8 --grad_accum 16 --generation_batch_size 512 --no_grad_ckpt
run D_bs8_gen1024   --per_device_train_batch_size 8 --grad_accum 16 --generation_batch_size 1024 --no_grad_ckpt
pip install -q liger-kernel 2>&1 | tail -1
run E_liger_bs8     --per_device_train_batch_size 8 --grad_accum 16 --generation_batch_size 512 --no_grad_ckpt --liger
run F_liger_bs16    --per_device_train_batch_size 16 --grad_accum 8 --generation_batch_size 1024 --no_grad_ckpt --liger
echo MATRIX_DONE
