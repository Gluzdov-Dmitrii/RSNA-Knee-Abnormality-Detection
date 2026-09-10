#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260910T0425Z-s22-resnet18-head
GPU=$(python3 -c "import json; print(json.load(open('$RUN/queue.request.json'))['gpus'][0])")
export CUDA_VISIBLE_DEVICES="$GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
bash "$PROJECT/training/launch_s22_head.sh"
