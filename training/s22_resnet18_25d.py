"""S22: ImageNet ResNet-18 2.5D, three independent plane heads, 12-target fusion.

Trains on PIXEL_CACHE_V1 (6 slots x 9 slices x 224). Each plane sees fluid+struct
as a 6-channel image. Hidden test is still decoded live on Kaggle; this is the
local/NSU pixel path, not DINO_CACHE_V1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "ops" / "tools" / "pixel_cache_v1.py").is_file() else HERE
for candidate in (HERE, ROOT / "ops" / "tools"):
    if (candidate / "pixel_cache_v1.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pixel_cache_v1 import TARGETS, PixelCacheV1  # noqa: E402
from pixel_dataset import KneePixelDataset, build_train_table  # noqa: E402

SEED = 2026
QUEUE_PY = "/home/scientists/gluz_d_s/kaggle/_control/resource_queue.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(n_targets: int = 12, pretrained: bool = True, dropout: float = 0.2):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    class PlaneHead(nn.Module):
        def __init__(self):
            super().__init__()
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            trunk = resnet18(weights=weights)
            conv1 = trunk.conv1
            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = conv1.weight
                new_conv.weight[:, 3:6] = conv1.weight
                new_conv.weight *= 0.5
            trunk.conv1 = new_conv
            trunk.fc = nn.Identity()
            self.trunk = trunk

        def forward(self, x):
            return self.trunk(x)

    class S22ResNet18(nn.Module):
        def __init__(self, n_targets: int = 12, dropout: float = 0.2):
            super().__init__()
            self.planes = nn.ModuleList([PlaneHead() for _ in range(3)])
            self.dropout = nn.Dropout(dropout)
            self.fusion = nn.Linear(3 * 512, n_targets)

        def forward(self, x, plane_mask):
            feats = []
            for plane, head in enumerate(self.planes):
                feat = head(x[:, plane])
                feats.append(feat * plane_mask[:, plane].unsqueeze(1))
            return self.fusion(self.dropout(torch.cat(feats, dim=1)))

    return S22ResNet18(n_targets=n_targets, dropout=dropout)


def masked_bce_with_logits(logits, y, labeled, pos_weight=None):
    import torch.nn.functional as F

    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    denom = labeled.sum().clamp_min(1.0)
    return (loss * labeled).sum() / denom


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray, labeled: np.ndarray) -> list[float | None]:
    from sklearn.metrics import roc_auc_score

    scores: list[float | None] = []
    for col in range(y_true.shape[1]):
        mask = labeled[:, col] > 0.5
        yt = (y_true[mask, col] >= 0.5).astype(np.int32)
        if mask.sum() < 8 or len(np.unique(yt)) < 2:
            scores.append(None)
            continue
        scores.append(float(roc_auc_score(yt, y_pred[mask, col])))
    return scores


def macro_auc(scores: list[float | None]) -> float | None:
    values = [s for s in scores if s is not None]
    if not values:
        return None
    return float(np.mean(values))


class Heartbeat:
    def __init__(self, run_id: str, token: str, interval: float = 55.0):
        self.run_id = run_id
        self.token = token
        self.interval = interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.wait(self.interval):
            try:
                subprocess.run(
                    [
                        "python3",
                        QUEUE_PY,
                        "heartbeat",
                        "--id",
                        self.run_id,
                        "--token",
                        self.token,
                    ],
                    check=False,
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                continue

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)


def torch_collate(batch: list[dict]) -> dict:
    import torch

    return {
        "uid": [item["uid"] for item in batch],
        "x": torch.from_numpy(np.stack([item["x"] for item in batch], axis=0)),
        "y": torch.from_numpy(np.stack([item["y"] for item in batch], axis=0)),
        "labeled": torch.from_numpy(np.stack([item["labeled"] for item in batch], axis=0)),
        "plane_mask": torch.from_numpy(np.stack([item["plane_mask"] for item in batch], axis=0)),
    }


def make_loader(dataset: KneePixelDataset, batch_size: int, shuffle: bool, workers: int):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=torch_collate,
        pin_memory=True,
        drop_last=False,
    )


def pos_weight_from_table(table: pd.DataFrame) -> np.ndarray:
    weights = []
    for name in TARGETS:
        values = table[name].to_numpy(dtype=np.float64)
        labeled = np.isfinite(values)
        pos = np.sum((values >= 0.5) & labeled)
        neg = np.sum((values < 0.5) & labeled)
        weights.append(1.0 if pos == 0 else float(np.clip(neg / max(pos, 1), 0.25, 8.0)))
    return np.asarray(weights, dtype=np.float32)


def apply_mixup(x, y, labeled, plane_mask, alpha: float):
    import torch

    if alpha <= 0 or x.size(0) < 2:
        return x, y, labeled, plane_mask
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1.0 - lam) * x[index]
    y = lam * y + (1.0 - lam) * y[index]
    labeled = torch.maximum(labeled, labeled[index])
    plane_mask = torch.maximum(plane_mask, plane_mask[index])
    return x, y, labeled, plane_mask


def run_epoch(model, loader, device, optimizer=None, scaler=None, pos_weight=None, mixup_alpha: float = 0.0):
    import torch

    train = optimizer is not None
    model.train(train)
    losses = []
    logits_all = []
    y_all = []
    labeled_all = []
    uids = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        labeled = batch["labeled"].to(device, non_blocking=True)
        plane_mask = batch["plane_mask"].to(device, non_blocking=True)
        if train:
            x, y, labeled, plane_mask = apply_mixup(x, y, labeled, plane_mask, mixup_alpha)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x, plane_mask)
                loss = masked_bce_with_logits(logits, y, labeled, pos_weight=pos_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().float().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
        labeled_all.append(labeled.detach().cpu().numpy())
        uids.extend(batch["uid"])
    pred = 1.0 / (1.0 + np.exp(-np.concatenate(logits_all, axis=0)))
    y_np = np.concatenate(y_all, axis=0)
    lab = np.concatenate(labeled_all, axis=0)
    scores = roc_auc(y_np, pred, lab)
    return {
        "loss": float(np.mean(losses)),
        "macro_auc": macro_auc(scores),
        "per_target_auc": {name: scores[i] for i, name in enumerate(TARGETS)},
        "pred": pred,
        "y": y_np,
        "labeled": lab,
        "uid": uids,
    }


def make_model(model_fn, args, pretrained: bool):
    dropout = float(getattr(args, "dropout", 0.2))
    try:
        return model_fn(pretrained=pretrained, dropout=dropout)
    except TypeError:
        return model_fn(pretrained=pretrained)


def train_folds(args, model_fn=None, experiment_key: str = "S22_RESNET18_25D") -> dict:
    import torch

    model_fn = model_fn or build_model
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cache = PixelCacheV1(Path(args.cache) if args.cache else None)
    table = build_train_table(cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = [int(part) for part in args.folds.split(",") if part != ""]
    weight_decay = float(getattr(args, "weight_decay", 1e-4))
    mixup_alpha = float(getattr(args, "mixup", 0.0))
    backbone_lr_mult = float(getattr(args, "backbone_lr_mult", 1.0))
    history = []
    oof_rows = []
    for fold in folds:
        fold_dir = out / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_table = table.loc[table["fold"] != fold].reset_index(drop=True)
        val_table = table.loc[table["fold"] == fold].reset_index(drop=True)
        train_ds = KneePixelDataset(cache, train_table, augment=True, seed=args.seed + fold)
        val_ds = KneePixelDataset(cache, val_table, augment=False, seed=args.seed)
        train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
        val_loader = make_loader(val_ds, args.batch_size, False, max(0, args.workers // 2))
        model = make_model(model_fn, args, pretrained=not args.no_pretrained).to(device)
        backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fusion")]
        head_params = list(model.fusion.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr * backbone_lr_mult},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        pos_w = torch.from_numpy(pos_weight_from_table(train_table)).to(device)
        best_auc = -1.0
        best_path = fold_dir / "best.pt"
        patience = 0
        fold_hist = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_stats = run_epoch(
                model, train_loader, device, optimizer, scaler, pos_w, mixup_alpha=mixup_alpha
            )
            scheduler.step()
            val_stats = run_epoch(model, val_loader, device)
            val_auc = val_stats["macro_auc"] if val_stats["macro_auc"] is not None else -1.0
            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_macro_auc": train_stats["macro_auc"],
                "val_loss": val_stats["loss"],
                "val_macro_auc": val_stats["macro_auc"],
                "seconds": round(time.time() - t0, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            fold_hist.append(row)
            print(json.dumps(row), flush=True)
            ckpt = {
                "fold": fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "val_macro_auc": val_auc,
                "per_target_auc": val_stats["per_target_auc"],
                "seed": args.seed,
                "targets": TARGETS,
            }
            torch.save(ckpt, fold_dir / "last.pt")
            if val_auc > best_auc:
                best_auc = val_auc
                patience = 0
                torch.save(ckpt, best_path)
                np.savez_compressed(
                    fold_dir / "val_pred.npz",
                    uid=np.asarray(val_stats["uid"]),
                    pred=val_stats["pred"],
                    y=val_stats["y"],
                    labeled=val_stats["labeled"],
                )
            else:
                patience += 1
                if patience >= args.patience:
                    break
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])
        val_stats = run_epoch(model, val_loader, device)
        for i, uid in enumerate(val_stats["uid"]):
            rec = {"StudyInstanceUID": uid, "fold": fold}
            for j, name in enumerate(TARGETS):
                rec[name] = float(val_stats["pred"][i, j])
            oof_rows.append(rec)
        history.append(
            {
                "fold": fold,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "best_val_macro_auc": best.get("val_macro_auc"),
                "per_target_auc": best.get("per_target_auc"),
                "epochs": fold_hist,
            }
        )
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.DataFrame(oof_rows)
    oof_path = out / "oof.csv"
    oof.to_csv(oof_path, index=False)
    merged = table[["StudyInstanceUID"] + TARGETS + ["fold"]].merge(
        oof, on="StudyInstanceUID", suffixes=("_y", "_p"), how="inner"
    )
    y = merged[[f"{name}_y" for name in TARGETS]].to_numpy(dtype=np.float32)
    p = merged[[f"{name}_p" for name in TARGETS]].to_numpy(dtype=np.float32)
    labeled = np.isfinite(y).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    oof_scores = roc_auc(y, p, labeled)
    summary = {
        "key": experiment_key,
        "checked_at_utc": utc_now(),
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "cache_root": str(cache.root),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": weight_decay,
        "dropout": float(getattr(args, "dropout", 0.2)),
        "mixup": mixup_alpha,
        "backbone_lr_mult": backbone_lr_mult,
        "pretrained": not args.no_pretrained,
        "in_channels": 6,
        "folds": folds,
        "oof_macro_auc": macro_auc(oof_scores),
        "oof_per_target_auc": {name: oof_scores[i] for i, name in enumerate(TARGETS)},
        "fold_results": [
            {k: v for k, v in item.items() if k != "epochs"} | {"n_epochs_ran": len(item["epochs"])}
            for item in history
        ],
        "history": history,
        "oof_csv": str(oof_path),
        "n_params": int(sum(p.numel() for p in make_model(model_fn, args, pretrained=False).parameters())),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def smoke_numpy(n: int = 8) -> dict:
    cache = PixelCacheV1()
    table = build_train_table(cache)
    dataset = KneePixelDataset(cache, table, fold=0, holdout=False, augment=False)
    batch = [dataset[i] for i in range(min(n, len(dataset)))]
    x = np.stack([item["x"] for item in batch], axis=0)
    y = np.stack([item["y"] for item in batch], axis=0)
    return {
        "n_train_fold0": len(dataset),
        "batch_x": list(x.shape),
        "batch_y": list(y.shape),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "cache": str(cache.root),
        "torch": False,
    }


def smoke_torch(n: int = 8) -> dict:
    import torch

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
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0)
    parser.add_argument("--experiment-key", type=str, default="S22_RESNET18_25D")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=SEED)
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
            result = train_folds(args, experiment_key=args.experiment_key)
        finally:
            if hb is not None:
                hb.close()
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
        return
    try:
        result = smoke_torch()
    except ImportError:
        result = smoke_numpy()
        result["note"] = "torch/torchvision missing; numpy loader smoke only"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
