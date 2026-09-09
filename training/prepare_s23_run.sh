#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260909T0635Z-s23-convnext-tiny-25d
mkdir -p "$RUN/logs" "$RUN/checkpoints" "$RUN/tmp"
echo rsna-s23-cv-20260909T0635Z > "$RUN/queue.id"
python3 -c 'import uuid; print(uuid.uuid4())' > "$RUN/queue.token"
chmod 600 "$RUN/queue.token"
echo RUN_READY
ls -ld "$RUN"
