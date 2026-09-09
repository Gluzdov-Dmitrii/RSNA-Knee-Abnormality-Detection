#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260909T0635Z-s23-convnext-tiny-25d
CODE=$PROJECT/code/pixel-cache-v1
GPU=$(python3 -c "import json; print(json.load(open('$RUN/queue.request.json'))['gpus'][0])")
export CUDA_VISIBLE_DEVICES="$GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
bash "$CODE/start_s23_job.sh"
