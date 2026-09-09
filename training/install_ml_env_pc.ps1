# Host-specific torch env on nsu-pc. Do not mutate py3.12-stdlib-v1.
$ErrorActionPreference = "Stop"
$Project = "C:\Users\User\kaggle\projects\rsna-knee-abnormality-detection"
$EnvDir = Join-Path $Project "envs\desktop-7t0uo8i\py3.12-torch-cu124-v1"
$PipCache = Join-Path $Project "cache\pip"
$TorchHome = Join-Path $Project "cache\torch"
$Tmp = Join-Path $Project "cache\tmp"
New-Item -ItemType Directory -Force -Path $PipCache, $TorchHome, $Tmp, (Split-Path $EnvDir) | Out-Null
$env:PIP_CACHE_DIR = $PipCache
$env:TORCH_HOME = $TorchHome
$env:TEMP = $Tmp
$env:TMP = $Tmp
$PyLauncher = Get-Command py -ErrorAction Stop
if (-not (Test-Path (Join-Path $EnvDir "Scripts\python.exe"))) {
    & $PyLauncher.Source -3.12 -m venv $EnvDir
}
$Py = Join-Path $EnvDir "Scripts\python.exe"
& $Py -m pip install --upgrade pip
& $Py -m pip install numpy pandas scikit-learn
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& $Py -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('vision', torchvision.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
& $Py -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); print('imagenet_ok')"
& $Py -m pip freeze | Set-Content -Encoding utf8 (Join-Path $EnvDir "pip-freeze.txt")
Write-Host "INSTALL_DONE host=desktop-7t0uo8i"
