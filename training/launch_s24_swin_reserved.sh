#!/bin/bash
set -euo pipefail
STAMP="${STAMP:?set STAMP like 20260910T1200Z}"
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/${STAMP}-s24-swin3d-t
GPU=$(python3 -c "import json; print(json.load(open('$RUN/queue.request.json'))['gpus'][0])")
export CUDA_VISIBLE_DEVICES="$GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
STAMP="$STAMP" bash "$PROJECT/training/launch_s24_swin.sh"
