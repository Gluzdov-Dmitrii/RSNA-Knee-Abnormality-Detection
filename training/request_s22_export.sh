#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260909T0735Z-s22-export
QUEUE=/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py
mkdir -p "$RUN/logs"
if [[ ! -f "$RUN/queue.id" ]]; then
  echo "rsna-s22-export-20260909T0735Z" > "$RUN/queue.id"
fi
if [[ ! -f "$RUN/queue.token" ]]; then
  python3 -c "import uuid; print(uuid.uuid4())" > "$RUN/queue.token"
  chmod 600 "$RUN/queue.token"
fi
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
  --cpu 4 \
  --ram-gib 16 \
  --disk-gib 5 \
  --minutes 30 > "$RUN/queue.request.json"
chmod 600 "$RUN/queue.request.json"
python3 - <<'PY'
import json
from pathlib import Path
run = Path("/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection/runs/20260909T0735Z-s22-export")
d = json.loads((run / "queue.request.json").read_text())
keys = ("id", "state", "health", "host", "alias", "pool", "gpus", "waiting_ahead")
print(json.dumps({k: d.get(k) for k in keys}, indent=2))
PY
