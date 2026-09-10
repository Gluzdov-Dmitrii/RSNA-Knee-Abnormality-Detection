#!/bin/bash
set -euo pipefail
STAMP="${STAMP:?set STAMP like 20260910T1200Z}"
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/${STAMP}-s24-swin3d-t
CODE=$PROJECT/code/pixel-cache-v1
HOST=$(hostname -s)
if [[ "$HOST" == "ngpu01" ]]; then
  PY=$PROJECT/envs/ngpu01/py3.11-torch-cu124-v1/bin/python
  BATCH="${BATCH:-8}"
else
  PY=$PROJECT/envs/prepost/py3.11-torch-cu124-v1/bin/python
  BATCH="${BATCH:-4}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_HOME=$PROJECT/cache/torch
export HF_HUB_CACHE=$PROJECT/cache/hf
export HF_HOME=$PROJECT/cache/hf
export TMPDIR=/tmp/rsna-s24-swin
mkdir -p "$RUN/logs" "$RUN/checkpoints" "$TMPDIR" "$HF_HUB_CACHE"
cd "$RUN"
if [[ -f "$RUN/train.pid" ]]; then
  OLD=$(cat "$RUN/train.pid")
  if [[ -d "/proc/$OLD" ]]; then
    echo "already running PID=$OLD"
    exit 0
  fi
fi
nohup "$PY" -u "$CODE/s24_swin3d_t.py" --train \
  --epochs 16 --batch-size "$BATCH" --lr 1e-4 --workers 4 --patience 5 \
  --dropout 0.25 --weight-decay 3e-4 --mixup 0.2 --backbone-lr-mult 0.3 \
  --layout vol9 --experiment-key S24_SWIN3D_T_9SLICE \
  --folds 0,1,2,3,4 --cache "$PROJECT/data/rsna-knee-uint8-224-9-c130" --out "$RUN/checkpoints" \
  --queue-id "$(cat "$RUN/queue.id")" --queue-token-file "$RUN/queue.token" \
  >>"$RUN/logs/train.log" 2>&1 &
echo $! > "$RUN/train.pid"
sleep 1
PID=$(cat "$RUN/train.pid")
START=$(awk '{print $22}' /proc/$PID/stat)
echo "$START" > "$RUN/train.start"
echo "HOST=$HOST PY=$PY BATCH=$BATCH"
echo "PID=$PID"
echo "START=$START"
head -n 40 "$RUN/logs/train.log" || true
