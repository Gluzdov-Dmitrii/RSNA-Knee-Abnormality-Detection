#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260909T0635Z-s23-convnext-tiny-25d
QUEUE=/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py
ID=$(cat "$RUN/queue.id")
TOKEN=$(cat "$RUN/queue.token")
python3 "$QUEUE" request \
  --id "$ID" \
  --owner rsna-agent \
  --token "$TOKEN" \
  --project rsna-knee-abnormality-detection \
  --run-path "$RUN" \
  --pool a100 \
  --count 1 \
  --cpu 8 \
  --ram-gib 32 \
  --disk-gib 15 \
  --minutes 120 > "$RUN/queue.request.json"
chmod 600 "$RUN/queue.request.json"
python3 - <<'PY'
import json
from pathlib import Path
run = Path("/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection/runs/20260909T0635Z-s23-convnext-tiny-25d")
d = json.loads((run / "queue.request.json").read_text())
keys = ("id", "state", "health", "host", "alias", "pool", "gpus", "waiting_ahead")
print(json.dumps({k: d.get(k) for k in keys}, indent=2))
PY
