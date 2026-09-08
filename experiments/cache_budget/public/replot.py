import matplotlib

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "ops" / "assets" / "CACHE_BUDGET_V1" / "cache_budget_v2_metrics.csv"
OUT = ROOT / "ops" / "assets" / "CACHE_BUDGET_V1"
STEVEN_GIB = 4407 * 6 * 9 * 224 * 224 / (1024**3)


def main() -> None:
    out = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    families = [
        ("resolution", "o-", "resolution, 9 slices, crop 130 mm"),
        ("slices", "s--", "slice count, 224², crop 130 mm"),
        ("tiny", "^:", "tiny, 3 slices, crop 130 mm"),
    ]
    for fam, fmt, label in families:
        d = out[out["family"] == fam].copy()
        if fam == "slices":
            extra = out[out["id"] == "res_224x9_c130"].copy()
            if not extra.empty:
                extra["id"] = "slc_224x9_c130"
                d = pd.concat([d, extra], ignore_index=True)
        d = d.sort_values("full_corpus_gib")
        yerr = [d["gbdt_macro_auc"] - d["gbdt_ci95_lo"], d["gbdt_ci95_hi"] - d["gbdt_macro_auc"]]
        ax.errorbar(d["full_corpus_gib"], d["gbdt_macro_auc"], yerr=yerr, fmt=fmt, label=label, capsize=3)
        for _, r in d.iterrows():
            ax.annotate(r["id"].replace("_c130", ""), (r["full_corpus_gib"], r["gbdt_macro_auc"]), fontsize=7)
    ax.axvline(STEVEN_GIB, color="0.4", ls=":", label="Steven 11.1 GiB design")
    ax.set_xlabel("Full-corpus cache size (GiB, uint8)")
    ax.set_ylabel("Subset OOF macro AUC (GBDT slot stats)")
    ax.set_title("Quality vs size  ·  crop 130 mm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    crop = pd.concat([out[out["id"] == "res_224x9_c130"], out[out["family"] == "crop"]]).drop_duplicates("id")
    crop = crop.sort_values("crop_mm")
    ax.bar([str(int(v)) for v in crop["crop_mm"]], crop["gbdt_macro_auc"], color="#4c78a8")
    for i, (_, r) in enumerate(crop.iterrows()):
        ax.plot([i, i], [r["gbdt_ci95_lo"], r["gbdt_ci95_hi"]], color="black")
        ax.scatter([i], [r["gbdt_macro_auc"]], color="black", zorder=3)
    ax.set_xlabel("Physical crop (mm) at 224² × 9, same GiB")
    ax.set_ylabel("Subset OOF macro AUC")
    ax.set_title("Crop is not a size knob")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cache_budget_v2_curve.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(7.5, 4.6))
    for fam, fmt, label in [
        ("resolution", "o-", "resolution, 9 slices"),
        ("slices", "s--", "slice count, 224²"),
        ("tiny", "^:", "tiny, 3 slices"),
    ]:
        d = out[out["family"] == fam].copy()
        if fam == "slices":
            extra = out[out["id"] == "res_224x9_c130"].copy()
            if not extra.empty:
                extra["id"] = "slc_224x9_c130"
                d = pd.concat([d, extra], ignore_index=True)
        d = d.sort_values("full_corpus_gib")
        ax.plot(d["full_corpus_gib"], d["ssim_vs_336x9_c130"], fmt, label=label)
        for _, r in d.iterrows():
            ax.annotate(r["id"].replace("_c130", ""), (r["full_corpus_gib"], r["ssim_vs_336x9_c130"]), fontsize=7)
    ax.set_xlabel("Full-corpus cache size (GiB, uint8)")
    ax.set_ylabel("Mean SSIM vs 336² × 9 crop 130 mm")
    ax.set_title("Fidelity to the densest crop-130 cache, not to raw DICOM")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "cache_budget_v2_ssim.png", dpi=150)
    print("wrote", OUT / "cache_budget_v2_curve.png")
    print("wrote", OUT / "cache_budget_v2_ssim.png")


if __name__ == "__main__":
    main()
