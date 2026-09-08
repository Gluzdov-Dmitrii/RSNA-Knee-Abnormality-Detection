#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260908T1625Z-s22-resnet18-25d
PY=$PROJECT/envs/prepost/py3.11-torch-cu124-v1/bin/python
CODE=$PROJECT/code/pixel-cache-v1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PIP_CACHE_DIR=$PROJECT/cache/pip
export TORCH_HOME=$PROJECT/cache/torch
export TMPDIR=$RUN/tmp
mkdir -p "$RUN/logs" "$RUN/checkpoints" "$TMPDIR"
TOKEN=$(cat "$RUN/queue.token")
QUEUE_ID=$(cat "$RUN/queue.id")
exec "$PY" -u "$CODE/s22_resnet18_25d.py" --train \
  --epochs 16 --batch-size 16 --lr 1e-4 --workers 2 --patience 5 \
  --folds 0,1,2,3,4 --out "$RUN/checkpoints" \
  --queue-id "$QUEUE_ID" --queue-token "$TOKEN" \
  >>"$RUN/logs/train.log" 2>&1
