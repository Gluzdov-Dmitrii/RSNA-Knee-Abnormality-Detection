#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260909T0735Z-s22-export
PY=$PROJECT/envs/ngpu01/py3.11-torch-cu124-v1/bin/python
SRC=$PROJECT/runs/20260908T1635Z-s22-resnet18-25d/checkpoints
OUT=$PROJECT/export/s22-resnet18-25d-folds
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export TMPDIR=/tmp/rsna-s22-export
mkdir -p "$RUN/logs" "$OUT" "$TMPDIR"
cd "$RUN"
nohup "$PY" -u "$PROJECT/training/export_s22_weights.py" --src "$SRC" --out "$OUT" \
  >>"$RUN/logs/export.log" 2>&1 &
echo $! > "$RUN/export.pid"
sleep 1
PID=$(cat "$RUN/export.pid")
START=$(awk '{print $22}' /proc/$PID/stat)
echo "$START" > "$RUN/export.start"
echo "PID=$PID"
echo "START=$START"
head -n 20 "$RUN/logs/export.log" || true
