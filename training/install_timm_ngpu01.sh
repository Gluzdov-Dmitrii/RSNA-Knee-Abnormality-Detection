#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
PY=$PROJECT/envs/ngpu01/py3.11-torch-cu124-v1/bin/python
export PIP_CACHE_DIR=$PROJECT/cache/pip
export TMPDIR=/tmp
export TORCH_HOME=$PROJECT/cache/torch
export HF_HUB_CACHE=$PROJECT/cache/hf
$PY -m pip install "timm==1.0.19"
$PY -c "import timm; m=timm.create_model('convnext_tiny', pretrained=False, num_classes=0); print('timm', timm.__version__, 'feat', m.num_features, 'params', sum(p.numel() for p in m.parameters()))"
$PY -m pip freeze > "$PROJECT/envs/ngpu01/py3.11-torch-cu124-v1/pip-freeze.txt"
echo TIMM_OK
