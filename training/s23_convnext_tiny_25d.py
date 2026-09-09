"""S23: timm ConvNeXt-Tiny 2.5D, three independent plane heads, 12-target fusion.

Same PIXEL_CACHE_V1 sampler and training loop as S22. CoAtNet is not allowed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from s22_resnet18_25d import (  # noqa: E402
    Heartbeat,
    masked_bce_with_logits,
    smoke_numpy,
    torch_collate,
    train_folds,
)
from pixel_cache_v1 import PixelCacheV1  # noqa: E402
from pixel_dataset import KneePixelDataset, build_train_table  # noqa: E402


def build_model(n_targets: int = 12, pretrained: bool = True):
    import timm
    import torch
    from torch import nn

    def expand_stem_to_6ch(module: nn.Module) -> None:
        for parent in module.modules():
            for name, child in parent.named_children():
                if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                    new = nn.Conv2d(
                        6,
                        child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=1,
                        bias=child.bias is not None,
                    )
                    with torch.no_grad():
                        new.weight[:, :3] = child.weight
                        new.weight[:, 3:6] = child.weight
                        new.weight *= 0.5
                        if child.bias is not None:
                            new.bias.copy_(child.bias)
                    setattr(parent, name, new)
                    return
        raise RuntimeError("no 3-channel conv found to expand")

    class PlaneHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                num_classes=0,
            )
            expand_stem_to_6ch(self.backbone)
            self.feat_dim = int(getattr(self.backbone, "num_features", 768))

        def forward(self, x):
            return self.backbone(x)

    class S23ConvNeXtTiny(nn.Module):
        def __init__(self, n_targets: int = 12):
            super().__init__()
            self.planes = nn.ModuleList([PlaneHead() for _ in range(3)])
            feat_dim = self.planes[0].feat_dim
            self.dropout = nn.Dropout(0.2)
            self.fusion = nn.Linear(3 * feat_dim, n_targets)

        def forward(self, x, plane_mask):
            feats = []
            for plane, head in enumerate(self.planes):
                feat = head(x[:, plane])
                feats.append(feat * plane_mask[:, plane].unsqueeze(1))
            return self.fusion(self.dropout(torch.cat(feats, dim=1)))

    return S23ConvNeXtTiny(n_targets=n_targets)


def smoke_torch(n: int = 8) -> dict:
    import torch

    from pixel_cache_v1 import PixelCacheV1
    from pixel_dataset import KneePixelDataset, build_train_table

    payload = smoke_numpy(n)
    cache = PixelCacheV1()
    table = build_train_table(cache)
    dataset = KneePixelDataset(cache, table, fold=0, holdout=False, augment=False)
    batch = torch_collate([dataset[i] for i in range(min(n, len(dataset)))])
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(batch["x"], batch["plane_mask"])
        loss = masked_bce_with_logits(logits, batch["y"], batch["labeled"])
    payload.update(
        {
            "torch": True,
            "cuda": bool(torch.cuda.is_available()),
            "logits": list(logits.shape),
            "loss": float(loss),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "arch": "convnext_tiny",
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--queue-id", type=str, default="")
    parser.add_argument("--queue-token", type=str, default="")
    args = parser.parse_args()
    if args.train:
        if not args.out:
            raise SystemExit("--out is required for --train")
        hb = None
        if args.queue_id and args.queue_token:
            hb = Heartbeat(args.queue_id, args.queue_token)
            hb.start()
        try:
            result = train_folds(
                args,
                model_fn=build_model,
                experiment_key="S23_CONVNEXT_TINY_25D",
            )
        finally:
            if hb is not None:
                hb.close()
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
        return
    try:
        result = smoke_torch()
    except ImportError:
        result = smoke_numpy()
        result["import_error"] = "torch or timm missing"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
