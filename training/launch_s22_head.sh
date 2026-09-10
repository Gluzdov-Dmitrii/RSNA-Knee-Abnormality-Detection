#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260910T0425Z-s22-resnet18-head
PY=$PROJECT/envs/ngpu01/py3.11-torch-cu124-v1/bin/python
CODE=$PROJECT/code/pixel-cache-v1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_HOME=$PROJECT/cache/torch
export HF_HUB_CACHE=$PROJECT/cache/hf
export HF_HOME=$PROJECT/cache/hf
export TMPDIR=/tmp/rsna-s22-head
mkdir -p "$RUN/logs" "$RUN/checkpoints" "$TMPDIR" "$HF_HUB_CACHE"
cd "$RUN"
if [[ -f "$RUN/train.pid" ]]; then
  OLD=$(cat "$RUN/train.pid")
  if [[ -d "/proc/$OLD" ]]; then
    echo "already running PID=$OLD"
    exit 0
  fi
fi
nohup "$PY" -u "$CODE/s22_resnet18_25d.py" --train \
  --epochs 20 --batch-size 16 --lr 1e-4 --workers 4 --patience 6 \
  --dropout 0.25 --weight-decay 3e-4 --mixup 0.2 --backbone-lr-mult 0.3 \
  --fusion-hidden 512 \
  --experiment-key S22_RESNET18_25D_HEAD \
  --folds 0,1,2,3,4 --cache "$PROJECT/data/rsna-knee-uint8-224-9-c130" --out "$RUN/checkpoints" \
  --queue-id "$(cat "$RUN/queue.id")" --queue-token-file "$RUN/queue.token" \
  >>"$RUN/logs/train.log" 2>&1 &
echo $! > "$RUN/train.pid"
sleep 1
PID=$(cat "$RUN/train.pid")
START=$(awk '{print $22}' /proc/$PID/stat)
echo "$START" > "$RUN/train.start"
echo "PID=$PID"
echo "START=$START"
head -n 30 "$RUN/logs/train.log" || true
