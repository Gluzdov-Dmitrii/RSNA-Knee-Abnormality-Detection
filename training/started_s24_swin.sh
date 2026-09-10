#!/bin/bash
set -euo pipefail
STAMP="${STAMP:?set STAMP like 20260910T1200Z}"
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/${STAMP}-s24-swin3d-t
QUEUE=/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py
ID=$(cat "$RUN/queue.id")
TOKEN=$(cat "$RUN/queue.token")
PID=$(cat "$RUN/train.pid")
START=$(cat "$RUN/train.start")
python3 "$QUEUE" started --id "$ID" --token "$TOKEN" --pid "$PID" --process-start "$START" > "$RUN/queue.started.json"
chmod 600 "$RUN/queue.started.json"
python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("$RUN/queue.started.json").read_text())
print(json.dumps({k: d.get(k) for k in ("id", "state", "health", "host", "gpus", "process")}, indent=2))
PY
echo "PID=$PID START=$START"
