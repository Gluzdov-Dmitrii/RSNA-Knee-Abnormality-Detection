#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260909T0635Z-s23-convnext-tiny-25d
PY=$PROJECT/envs/ngpu01/py3.11-torch-cu124-v1/bin/python
CODE=$PROJECT/code/pixel-cache-v1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_HOME=$PROJECT/cache/torch
export HF_HUB_CACHE=$PROJECT/cache/hf
export HF_HOME=$PROJECT/cache/hf
export TMPDIR=/tmp/rsna-s23-cv
mkdir -p "$RUN/logs" "$RUN/checkpoints" "$TMPDIR" "$HF_HUB_CACHE"
TOKEN=$(cat "$RUN/queue.token")
QUEUE_ID=$(cat "$RUN/queue.id")
cd "$RUN"
nohup "$PY" -u "$CODE/s23_convnext_tiny_25d.py" --train \
  --epochs 16 --batch-size 16 --lr 1e-4 --workers 4 --patience 5 \
  --folds 0,1,2,3,4 --out "$RUN/checkpoints" \
  --queue-id "$QUEUE_ID" --queue-token "$TOKEN" \
  >>"$RUN/logs/train.log" 2>&1 &
echo $! > "$RUN/train.pid"
sleep 1
PID=$(cat "$RUN/train.pid")
START=$(awk '{print $22}' /proc/$PID/stat)
echo "PID=$PID"
echo "START=$START"
ls -l /proc/$PID/exe
head -n 40 "$RUN/logs/train.log" || true
