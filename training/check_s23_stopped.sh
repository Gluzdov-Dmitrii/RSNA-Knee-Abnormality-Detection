#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260909T0635Z-s23-convnext-tiny-25d
PID=$(cat "$RUN/train.pid")
if ps -p "$PID" > /dev/null 2>&1; then
  echo "STILL_RUNNING pid=$PID"
  ps -p "$PID" -o pid,etime,cmd
  exit 1
fi
pgrep -af s23_convnext_tiny_25d.py || echo "no s23 python"
echo STOPPED
