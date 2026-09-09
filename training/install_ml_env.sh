#!/bin/bash
set -euo pipefail
PROJECT=/home/scientists/gluz_d_s/kaggle/projects/rsna-knee-abnormality-detection
HOST=${1:-}
if [ -z "$HOST" ]; then
  case "$(hostname)" in
    prepost) HOST=prepost ;;
    ngpu01) HOST=ngpu01 ;;
    *)
      echo "usage: $0 prepost|ngpu01" >&2
      exit 1
      ;;
  esac
fi
ENV=$PROJECT/envs/$HOST/py3.11-torch-cu124-v1
export PIP_CACHE_DIR=$PROJECT/cache/pip
export TMPDIR=/tmp/rsna-pip-$HOST
export TORCH_HOME=$PROJECT/cache/torch
export XDG_CACHE_HOME=$PROJECT/cache/xdg
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$ENV"
if [ ! -x "$ENV/bin/python" ]; then
  python3 -m venv --without-pip "$ENV"
fi
if [ ! -x "$ENV/bin/pip" ]; then
  python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '$PROJECT/cache/get-pip.py')"
  "$ENV/bin/python" "$PROJECT/cache/get-pip.py"
fi
PY=$ENV/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install numpy pandas scikit-learn
$PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
$PY -c "import torch, torchvision; print('host', '$HOST'); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('vision', torchvision.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
$PY -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); print('imagenet_ok')"
$PY -m pip freeze > "$ENV/pip-freeze.txt"
echo INSTALL_DONE host=$HOST
