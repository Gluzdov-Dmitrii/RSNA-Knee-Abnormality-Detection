#!/bin/bash
set -euo pipefail
RUN=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection/runs/20260909T0635Z-s23-convnext-tiny-25d
PID=$(cat "$RUN/train.pid")
awk '{print $22}' /proc/$PID/stat > "$RUN/train.start"
echo "PID=$PID"
echo -n "START="
cat "$RUN/train.start"
ps -p "$PID" -o pid,etime,cmd
