"""S22 skeleton: ResNet-18 2.5D on PIXEL_CACHE_V1.

Three independent plane trunks (sagittal / coronal / axial fluid slots),
each seeing slices 1/4/7 as RGB, then a linear 12-target fusion.

This is the pixel-cache training path. It does not use DINO_CACHE_V1.
Hidden test is still decoded live in the Kaggle scoring notebook.

Requires torch + torchvision in the selected env. The local kg venv and the
NSU stdlib baseline do not have them yet. Do not start a GPU job without
resource_queue.py reservation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ops" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from pixel_cache_v1 import TARGETS, PixelCacheV1  # noqa: E402
from pixel_dataset import KneePixelDataset, build_train_table, numpy_collate  # noqa: E402


def build_model():
    import torch
    from torch import nn
    from torchvision.models import resnet18

    class PlaneHead(nn.Module):
        def __init__(self):
            super().__init__()
            trunk = resnet18(weights=None)
            trunk.fc = nn.Identity()
            self.trunk = trunk

        def forward(self, x):
            return self.trunk(x)

    class S22ResNet18(nn.Module):
        def __init__(self, n_targets: int = 12):
            super().__init__()
            self.planes = nn.ModuleList([PlaneHead() for _ in range(3)])
            self.fusion = nn.Linear(3 * 512, n_targets)

        def forward(self, x, plane_mask):
            # x: (B, 3, 3, 224, 224); plane_mask: (B, 3)
            feats = []
            for plane, head in enumerate(self.planes):
                feat = head(x[:, plane])
                feats.append(feat * plane_mask[:, plane].unsqueeze(1))
            return self.fusion(torch.cat(feats, dim=1))

    return S22ResNet18(n_targets=len(TARGETS))


def masked_bce_with_logits(logits, y, labeled):
    import torch.nn.functional as F

    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    denom = labeled.sum().clamp_min(1.0)
    return (loss * labeled).sum() / denom


def smoke_numpy(n: int = 8) -> dict:
    cache = PixelCacheV1()
    table = build_train_table(cache)
    dataset = KneePixelDataset(cache, table, fold=0, holdout=False)
    batch = numpy_collate([dataset[i] for i in range(min(n, len(dataset)))])
    return {
        "n_train_fold0": len(dataset),
        "batch_x": list(batch["x"].shape),
        "batch_y": list(batch["y"].shape),
        "x_min": float(batch["x"].min()),
        "x_max": float(batch["x"].max()),
        "labeled_frac": float(batch["labeled"].mean()),
        "torch": False,
    }


def smoke_torch(n: int = 8) -> dict:
    import torch

    payload = smoke_numpy(n)
    cache = PixelCacheV1()
    table = build_train_table(cache)
    dataset = KneePixelDataset(cache, table, fold=0, holdout=False)
    batch = numpy_collate([dataset[i] for i in range(min(n, len(dataset)))])
    model = build_model()
    model.eval()
    x = torch.from_numpy(batch["x"])
    y = torch.from_numpy(batch["y"])
    labeled = torch.from_numpy(batch["labeled"])
    plane_mask = torch.from_numpy(batch["plane_mask"])
    with torch.no_grad():
        logits = model(x, plane_mask)
        loss = masked_bce_with_logits(logits, y, labeled)
    payload.update(
        {
            "torch": True,
            "logits": list(logits.shape),
            "loss": float(loss),
            "n_params": int(sum(p.numel() for p in model.parameters())),
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=0, help="GPU train is opt-in and requires --epochs > 0")
    args = parser.parse_args()
    if args.epochs > 0:
        raise SystemExit(
            "GPU training is not started from this skeleton. Reserve via "
            "resource_queue.py, then run a dedicated train command."
        )
    try:
        result = smoke_torch()
    except ImportError:
        result = smoke_numpy()
        result["note"] = "torch/torchvision missing; numpy loader smoke only"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
