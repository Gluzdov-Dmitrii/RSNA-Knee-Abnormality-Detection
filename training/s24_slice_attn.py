"""S24-attn: ImageNet ResNet-18 per slice, attention pool over 9 slices.

Same vol9 cache as S24 r3d_18. Each plane is 9 RGB-like frames
(fluid, struct, mean). Not a 3D conv and not a Raptor substitute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from s22_resnet18_25d import Heartbeat, train_folds  # noqa: E402


def build_model(n_targets: int = 12, pretrained: bool = True, dropout: float = 0.2, fusion_hidden: int = 0):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    class SliceAttnHead(nn.Module):
        def __init__(self):
            super().__init__()
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            trunk = resnet18(weights=weights)
            trunk.fc = nn.Identity()
            self.trunk = trunk
            self.attn = nn.Linear(512, 1)

        def forward(self, x):
            batch, channels, time, height, width = x.shape
            frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
            feat = self.trunk(frames).view(batch, time, 512)
            weights = torch.softmax(self.attn(feat), dim=1)
            return (feat * weights).sum(dim=1)

    class S24SliceAttn(nn.Module):
        def __init__(self, n_targets: int = 12, dropout: float = 0.2, fusion_hidden: int = 0):
            super().__init__()
            self.planes = nn.ModuleList([SliceAttnHead() for _ in range(3)])
            self.dropout = nn.Dropout(dropout)
            in_dim = 3 * 512
            hidden = int(fusion_hidden)
            if hidden > 0:
                self.fusion = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n_targets),
                )
            else:
                self.fusion = nn.Linear(in_dim, n_targets)

        def forward(self, x, plane_mask):
            feats = []
            for plane, head in enumerate(self.planes):
                feat = head(x[:, plane])
                feats.append(feat * plane_mask[:, plane].unsqueeze(1))
            return self.fusion(self.dropout(torch.cat(feats, dim=1)))

    return S24SliceAttn(n_targets=n_targets, dropout=dropout, fusion_hidden=fusion_hidden)


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
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fusion-hidden", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.3)
    parser.add_argument("--experiment-key", type=str, default="S24_SLICE_ATTN")
    parser.add_argument("--layout", type=str, default="vol9")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--queue-id", type=str, default="")
    parser.add_argument("--queue-token", type=str, default="")
    parser.add_argument("--queue-token-file", type=str, default="")
    args = parser.parse_args()
    if args.queue_token_file and not args.queue_token:
        args.queue_token = Path(args.queue_token_file).read_text(encoding="utf-8").strip()
    if args.train:
        if not args.out:
            raise SystemExit("--out is required for --train")
        hb = None
        if args.queue_id and args.queue_token:
            hb = Heartbeat(args.queue_id, args.queue_token)
            hb.start()
        try:
            result = train_folds(args, model_fn=build_model, experiment_key=args.experiment_key)
        finally:
            if hb is not None:
                hb.close()
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
        return
    print(json.dumps({"smoke": True, "layout": "vol9", "arch": "resnet18_slice_attn"}))


if __name__ == "__main__":
    main()
