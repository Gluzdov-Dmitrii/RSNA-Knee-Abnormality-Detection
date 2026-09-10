#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260910T0545Z-s24-r3d18-9slice
QUEUE=/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py
ID=$(cat "$RUN/queue.id")
TOKEN=$(cat "$RUN/queue.token")
python3 "$QUEUE" release --id "$ID" --token "$TOKEN" --verified-stopped > "$RUN/queue.release.json"
chmod 600 "$RUN/queue.release.json"
python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("$RUN/queue.release.json").read_text())
print(json.dumps({k: d.get(k) for k in ("id", "state", "health", "finished")}, indent=2))
PY
