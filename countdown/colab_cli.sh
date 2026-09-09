#!/usr/bin/env bash
# Drive the Countdown GRPO run through the official Colab CLI (no browser).
#   ./colab_cli.sh setup            # new A100 session + deps + upload the 4 .py files
#   ./colab_cli.sh train [ARGS...]  # launch train_countdown_grpo.py in the background on the VM (nohup, log file)
#   ./colab_cli.sh tail [N]         # last N lines of the training log
#   ./colab_cli.sh metrics          # parsed per-step metrics from the log
#   ./colab_cli.sh eval [ARGS...]   # run eval_countdown.py (foreground, streams)
#   ./colab_cli.sh fetch            # download results/ + curves + trainer_state to ./colab_out/
#   ./colab_cli.sh stop
# Env: SESSION (default cd-a100), COLAB (default ~/.local/bin/colab), EXEC_TIMEOUT (s, default 600).
# Known CLI bug (0.6.0): needs  uv tool install --force google-colab-cli --with "jupyter-kernel-client<1"
# Kernel state persists across `colab exec` calls, so setting WANDB_API_KEY/HF_TOKEN once via
#   echo "import os; os.environ['WANDB_API_KEY']='...'; os.environ['HF_TOKEN']='...'" | colab exec -s cd-a100
# is inherited by the training subprocess launched later.
set -euo pipefail
SESSION="${SESSION:-cd-a100}"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE=/content/countdown
LOG=$REMOTE/train.log

x() { "$COLAB" exec -s "$SESSION" --timeout "${EXEC_TIMEOUT:-600}"; }   # piped python -> kernel (default 30 s idle timeout is too short)

case "${1:-}" in
  setup)
    "$COLAB" new -s "$SESSION" --gpu A100
    "$COLAB" install -s "$SESSION" "trl>=1.11" "transformers>=5.2" datasets peft accelerate wandb matplotlib flash-linear-attention huggingface_hub
    echo "import os; os.makedirs('$REMOTE', exist_ok=True)" | x
    for f in data.py rewards.py train_countdown_grpo.py eval_countdown.py; do
      "$COLAB" upload -s "$SESSION" "$HERE/$f" "$REMOTE/$f"
    done
    echo "import subprocess; print(subprocess.run('cd $REMOTE && nvidia-smi --query-gpu=name,memory.total --format=csv && python -c \"from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print(\\'fla OK\\')\" && python rewards.py | tail -1', shell=True, capture_output=True, text=True).stdout)" | x
    ;;
  train)
    shift
    ARGS="$*"
    echo "import subprocess, os; os.chdir('$REMOTE'); p=subprocess.Popen('nohup python -u train_countdown_grpo.py $ARGS > $LOG 2>&1 &', shell=True); print('launched, log: $LOG')" | x
    ;;
  tail)
    N="${2:-40}"
    echo "print(open('$LOG').read()[-20000:].splitlines()[-$N:] and '\n'.join(open('$LOG').read().splitlines()[-$N:]))" | x
    ;;
  metrics)
    echo "import re
for l in open('$LOG'):
    if \"'step_time'\" in l:
        g=lambda k: (re.search(\"'%s': '([^']+)'\"%re.escape(k), l) or [None,'?'])[1]
        print('len',g('completions/mean_length'),'clip',g('completions/clipped_ratio'),'fmt',g('rewards/format_reward/mean'),'ans',g('rewards/answer_reward/mean'),'zstd',g('frac_reward_zero_std'),'ent',g('entropy'),'t',g('step_time'))" | x
    ;;
  eval)
    shift
    echo "import subprocess, os; os.chdir('$REMOTE'); subprocess.run('python -u eval_countdown.py $*', shell=True)" | x
    ;;
  fetch)
    mkdir -p "$HERE/colab_out"
    echo "import tarfile,glob,os; os.chdir('$REMOTE'); t=tarfile.open('/content/out.tgz','w:gz'); [t.add(p) for p in ['results']+glob.glob('runs/*/curves.png')+glob.glob('runs/*/checkpoint-*/trainer_state.json')+glob.glob('runs/*/final/trainer_state.json') if os.path.exists(p)]; t.close(); print(os.path.getsize('/content/out.tgz'))" | x
    "$COLAB" download -s "$SESSION" /content/out.tgz "$HERE/colab_out/out.tgz" && tar -xzf "$HERE/colab_out/out.tgz" -C "$HERE/colab_out" && ls -R "$HERE/colab_out" | head -30
    ;;
  vllm)   # optional: fast generation. Check torch/fla survive the install before using --use_vllm.
    "$COLAB" install -s "$SESSION" "vllm==0.27.1"
    echo "import subprocess; print(subprocess.run('python -c \"import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__); from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print(\'fla OK\')\"', shell=True, capture_output=True, text=True))" | x
    ;;
  py)     # arbitrary python in the kernel:  ./colab_cli.sh py "print(1)"
    shift; echo "$*" | x ;;
  stop)  "$COLAB" stop -s "$SESSION" ;;
  status) "$COLAB" status -s "$SESSION"; "$COLAB" sessions ;;
  *) sed -n '2,12p' "$0" ;;
esac
