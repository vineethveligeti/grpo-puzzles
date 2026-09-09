#!/usr/bin/env bash
# Keep the Countdown GRPO run alive across Colab VM deaths (CLI sessions were reclaimed at ~60 min, twice).
#   HUB_REPO=<user>/qwen3.5-0.8b-countdown-grpo [MAX_STEPS=800 RUN_NAME=... WANDB_RUN_ID=...] ./supervise.sh [extra train args]
#   e.g. stage 1b: MAX_STEPS=800 RUN_NAME=cd-0.8b-3only-s1b WANDB_RUN_ID=cd08b-3only-s1b ./supervise.sh --epsilon_high 0.28 --no_grad_ckpt --generation_batch_size 1024 --vllm_gpu_memory_utilization 0.45
# Loop: no live session -> new A100 + setup_vm.sh + tokens + launch train with --resume auto --hub_repo $HUB_REPO.
# Progress is read from the VM log every POLL seconds; done when runs/<out>/final exists (also pushed to the Hub).
# Needs in the LOCAL env: HF_TOKEN (Hub push), WANDB_API_KEY (curves). Log: supervise.log next to this script.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
: "${HUB_REPO:?set HUB_REPO=<user>/<repo>}"
SESSION="${SESSION:-cd-a100}"; COLAB="${COLAB:-$HOME/.local/bin/colab}"; REMOTE=/content/countdown
OUT=runs/countdown-qwen3.5-0.8b-base; POLL="${POLL:-300}"; MAX_RESTARTS="${MAX_RESTARTS:-8}"
MAX_STEPS="${MAX_STEPS:-400}"; RUN_NAME="${RUN_NAME:-cd-0.8b-base-3only-g16-vllm}"; WANDB_RUN_ID="${WANDB_RUN_ID:-cd08b-3only-g16-vllm}"
# extra args ($*) come last, so they override anything here (argparse keeps the last value)
TRAIN_ARGS="--model /content/qwen3.5-0.8b-text --output_dir $OUT --max_steps $MAX_STEPS --save_steps 50 --report_to wandb \
 --run_name $RUN_NAME --curriculum 3only --max_completion_length 1024 --per_device_train_batch_size 4 \
 --grad_accum 32 --num_generations 16 --generation_batch_size 512 --use_vllm --vllm_gpu_memory_utilization 0.35 \
 --resume auto --hub_repo $HUB_REPO $*"
log() { echo "$(date '+%F %T') $*" | tee -a supervise.log; }
x() { "$COLAB" exec -s "$SESSION" --timeout "${1:-120}"; }
alive() { "$COLAB" sessions </dev/null 2>&1 | grep -q "^\[$SESSION\]"; }
restarts=0
while :; do
  if ! alive; then
    [ "$restarts" -ge "$MAX_RESTARTS" ] && { log "giving up after $restarts restarts"; exit 1; }
    restarts=$((restarts+1)); log "no live session -> creating (restart #$restarts)"
    "$COLAB" new -s "$SESSION" --gpu A100 </dev/null 2>&1 | tail -2 | tee -a supervise.log || { sleep 120; continue; }
    echo "import os; os.makedirs('$REMOTE', exist_ok=True)" | x 60 >/dev/null
    for f in data.py rewards.py train_countdown_grpo.py eval_countdown.py make_text_only_ckpt.py setup_vm.sh; do
      "$COLAB" upload -s "$SESSION" "$HERE/$f" "$REMOTE/$f" </dev/null >/dev/null 2>&1
    done
    log "setup_vm.sh (deps + vllm + text-only ckpt) ..."
    echo "import subprocess; r=subprocess.run('bash $REMOTE/setup_vm.sh', shell=True); print('setup rc', r.returncode)" | x 1500 | tail -1 | tee -a supervise.log
    echo "import os; os.environ.update({k:v for k,v in {'WANDB_API_KEY':'${WANDB_API_KEY:-}','HF_TOKEN':'${HF_TOKEN:-}','WANDB_PROJECT':'grpo-countdown','WANDB_RESUME':'allow','WANDB_RUN_ID':'$WANDB_RUN_ID'}.items() if v}); print('tokens', {k:(k in os.environ) for k in ('WANDB_API_KEY','HF_TOKEN')})" | x 60 | tail -1 | tee -a supervise.log
    echo "import subprocess,os; os.chdir('$REMOTE'); subprocess.Popen('nohup python -u train_countdown_grpo.py $TRAIN_ARGS > train.log 2>&1 &', shell=True); print('launched')" | x 60 | tail -1 | tee -a supervise.log
  fi
  sleep "$POLL"
  alive || continue
  echo "import os,re,subprocess
p='$REMOTE/train.log'; s=open(p).read() if os.path.exists(p) else ''
prog=re.findall(r'(\d+)/400 \[', s); steps=[l for l in s.splitlines() if \"'completions/mean_length'\" in l]
g=lambda l,k: (re.search(r\"'%s': '([^']+)'\"%re.escape(k), l) or [None,'?'])[1]
last=steps[-1] if steps else ''
alive=subprocess.run('pgrep -f train_countdown_grpo',shell=True,capture_output=True).returncode==0
done=os.path.exists('$REMOTE/$OUT/final')
err=[l for l in s.splitlines() if 'Traceback' in l or 'Error' in l][-1:] 
print('PROGRESS step', prog[-1] if prog else '?', '| len', g(last,'completions/mean_length'), 'fmt', g(last,'rewards/format_reward/mean'), 'ans', g(last,'rewards/answer_reward/mean'), 'zstd', g(last,'frac_reward_zero_std'), '| proc', alive, '| done', done, '|', err)" | x 60 | grep PROGRESS | tee -a supervise.log
  if tail -1 supervise.log | grep -q "done True"; then log "TRAINING COMPLETE"; exit 0; fi
  if tail -1 supervise.log | grep -q "proc False | done False"; then log "process died on a live VM -> relaunching"; echo "import subprocess,os; os.chdir('$REMOTE'); subprocess.Popen('nohup python -u train_countdown_grpo.py $TRAIN_ARGS >> train.log 2>&1 &', shell=True); print('relaunched')" | x 60 | tail -1 | tee -a supervise.log; fi
done
