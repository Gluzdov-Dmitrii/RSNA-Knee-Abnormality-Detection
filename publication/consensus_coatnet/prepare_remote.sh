#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
RUN=$PROJECT/runs/20260910T0913Z-codex-consensus
CODE=$PROJECT/code/consensus-coatnet-v3
PY=$PROJECT/envs/ngpu01/py3.11-torch-cu124-v1/bin/python
test "$(hostname)" = ngpu01
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=$PROJECT/cache/consensus-coatnet-v1/hf
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$PROJECT/cache/consensus-coatnet-v1/torch
export XDG_CACHE_HOME=$PROJECT/cache/consensus-coatnet-v1/xdg
export TMPDIR=$RUN/tmp
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
"$PY" -m unittest discover -s "$CODE" -p test_contract.py -v
"$PY" -u "$CODE/prepare.py" --cache "$PROJECT/data/rsna-knee-uint8-224-9-c130" \
  --folds "$PROJECT/data/FOLDS_V1/folds.csv" --labels "$PROJECT/data/labels" \
  --gold-uids "$RUN/inputs/gold_uids.csv" --out "$RUN/prepared_correctnorm" --verify-pixels
