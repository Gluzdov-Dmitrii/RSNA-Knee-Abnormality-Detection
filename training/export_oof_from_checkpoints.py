"""Rebuild fold-out OOF predictions from exported fold checkpoints.

This is for local/NSU validation artifacts. It does not train and does not touch
Kaggle. Each checkpoint is evaluated only on its own FOLDS_V1 holdout fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "ops" / "tools" / "pixel_cache_v1.py").is_file() else HERE
for candidate in (HERE, ROOT / "ops" / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pixel_cache_v1 import TARGETS, PixelCacheV1, sha256_file  # noqa: E402
from pixel_dataset import KneePixelDataset, build_train_table  # noqa: E402
from s22_resnet18_25d import macro_auc, roc_auc, torch_collate  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def model_builder(arch: str):
    if arch == "s22_resnet18_25d":
        from s22_resnet18_25d import build_model

        return build_model, "rgb6"
    if arch == "s24_r3d18_9slice":
        from s24_r3d18_9slice import build_model

        return build_model, "vol9"
    if arch == "s24_r2plus1d_9slice":
        from s24_r2plus1d_18 import build_model

        return build_model, "vol9"
    if arch == "s24_swin3d_t_9slice":
        from s24_swin3d_t import build_model

        return build_model, "vol9"
    raise ValueError(f"unknown arch {arch}")


def load_state(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"{path} is not a state_dict checkpoint")


def make_loader(dataset: KneePixelDataset, batch_size: int, workers: int):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=torch_collate,
        pin_memory=True,
        drop_last=False,
    )


def predict(model, loader, device, max_batches: int | None = None) -> dict:
    import torch

    model.eval()
    logits_all = []
    y_all = []
    labeled_all = []
    uids = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["x"].to(device, non_blocking=True)
            plane_mask = batch["plane_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x, plane_mask)
            logits_all.append(logits.float().cpu().numpy())
            y_all.append(batch["y"].numpy())
            labeled_all.append(batch["labeled"].numpy())
            uids.extend(batch["uid"])
    logits_np = np.concatenate(logits_all, axis=0)
    pred = 1.0 / (1.0 + np.exp(-logits_np))
    return {
        "uid": uids,
        "pred": pred,
        "y": np.concatenate(y_all, axis=0),
        "labeled": np.concatenate(labeled_all, axis=0),
    }


def evaluate_oof(table: pd.DataFrame, oof: pd.DataFrame) -> dict:
    merged = table[["StudyInstanceUID", "fold"] + TARGETS].merge(
        oof, on="StudyInstanceUID", suffixes=("_y", "_p"), how="inner", validate="one_to_one"
    )
    fold_col = "fold" if "fold" in merged.columns else "fold_y"
    if "fold_y" in merged.columns and "fold_p" in merged.columns:
        if not (merged["fold_y"].to_numpy() == merged["fold_p"].to_numpy()).all():
            raise ValueError("OOF fold column does not match FOLDS_V1 fold assignments")
    y = merged[[f"{name}_y" for name in TARGETS]].to_numpy(dtype=np.float32)
    p = merged[[f"{name}_p" for name in TARGETS]].to_numpy(dtype=np.float32)
    labeled = np.isfinite(y).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    scores = roc_auc(y, p, labeled)
    return {
        "n_rows": int(len(merged)),
        "oof_macro_auc": macro_auc(scores),
        "oof_per_target_auc": {name: scores[i] for i, name in enumerate(TARGETS)},
        "fold_sizes": {str(k): int(v) for k, v in merged[fold_col].value_counts().sort_index().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=[
        "s22_resnet18_25d",
        "s24_r3d18_9slice",
        "s24_r2plus1d_9slice",
        "s24_swin3d_t_9slice",
    ])
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-pattern", default="fold{fold}.pt")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--fusion-hidden", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    build_model, layout = model_builder(args.arch)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cache = PixelCacheV1(args.cache)
    table = build_train_table(cache)
    folds = [int(part) for part in args.folds.split(",") if part]
    max_batches = args.max_batches if args.max_batches > 0 else None
    rows: list[dict] = []
    fold_receipts = []

    for fold in folds:
        ckpt_path = args.checkpoint_root / args.checkpoint_pattern.format(fold=fold)
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        t0 = time.time()
        val_table = table.loc[table["fold"] == fold].reset_index(drop=True)
        dataset = KneePixelDataset(cache, val_table, fold=None, holdout=False, augment=False, layout=layout)
        loader = make_loader(dataset, args.batch_size, args.workers)
        model = build_model(pretrained=False, dropout=args.dropout, fusion_hidden=args.fusion_hidden).to(device)
        missing, unexpected = model.load_state_dict(load_state(ckpt_path), strict=True)
        if missing or unexpected:
            raise RuntimeError(f"{ckpt_path} load drift missing={missing} unexpected={unexpected}")
        pred = predict(model, loader, device, max_batches=max_batches)
        for i, uid in enumerate(pred["uid"]):
            rec = {"StudyInstanceUID": uid, "fold": fold}
            for j, name in enumerate(TARGETS):
                rec[name] = float(pred["pred"][i, j])
            rows.append(rec)
        fold_receipts.append({
            "fold": fold,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "n_pred": len(pred["uid"]),
            "seconds": round(time.time() - t0, 2),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(fold_receipts[-1]), flush=True)

    oof = pd.DataFrame(rows)
    oof_path = args.out / "oof.csv"
    oof.to_csv(oof_path, index=False)
    metrics = evaluate_oof(table, oof)
    metrics.update({
        "key": f"{args.arch}_rebuilt_oof",
        "checked_at_utc": utc_now(),
        "device": str(device),
        "cache_root": str(cache.root),
        "cache_spec_sha256": sha256_file(cache.root / "SPEC.json"),
        "arch": args.arch,
        "layout": layout,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "folds": folds,
        "max_batches": max_batches,
        "oof_csv": str(oof_path),
        "fold_receipts": fold_receipts,
    })
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"oof_per_target_auc", "fold_receipts"}}, indent=2))


if __name__ == "__main__":
    main()
