"""Human-facing notebook; requires measured evidence and a confirmed Dataset receipt."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PUBLIC=HERE.parent/'public'
EVIDENCE=ROOT/'artifacts/cache_budget_recheck/evidence'
PUBLISHED=ROOT/'artifacts/cache_budget_recheck/dataset_publication.json'

def cell(kind,source,cid):
    value=dict(cell_type=kind,id=cid,source=source.strip()+'\n',metadata={})
    if kind=='code':
        value.update(execution_count=None,outputs=[])
        value['metadata']={'jupyter':{'source_hidden':True},'tags':['hide-input'],
                           'source_hidden':True,'hide_input':True,'collapsed':True,
                           '_kg_hide-input':True}
    return value

def conclusions(receipt):
    rows={r['id']:r for r in receipt['rows']}
    base=rows['res_224x9_c130']
    res=[rows[k] for k in rows if k.startswith('res_') and rows[k]['img']>=160]
    three=rows['slc_224x3_c130']
    more=rows['res_336x9_c130']
    significant_resolution=any(r['delta_ci95_lo']>.01 and r['holm_p']<.05 for r in res)
    resolution=('This verifier finds a clear gain from extra resolution; the earlier plateau claim does not hold.'
                if significant_resolution else
                '224² → 336²: 2.25× the storage for an observed +0.005 AUC; the gain is uncertain, not a proven flat plateau.')
    if three['delta_ci95_hi'] < -.01 and three['holm_p']<.05:
        slices='The cheap mistake was too few slices: three slices lost signal that more pixels did not recover.'
    elif three['delta_auc'] < 0:
        slices='Three slices scored lower, but this verifier did not establish a clear quality cliff. Treat the earlier slice-cliff claim as provisional.'
    else:
        slices='This verifier did not reproduce the earlier three-slice quality cliff. Keep nine slices as a conservative geometry choice, not a proven optimum.'
    crop=rows['crp_224x9_c160']
    crop_text=(f"The 160 mm crop changed AUC by {crop['delta_auc']:+.3f} versus 130 mm "
               f"(paired 95% interval {crop['delta_ci95_lo']:+.3f} to {crop['delta_ci95_hi']:+.3f}). ")
    if crop['delta_ci95_lo']>.01 and crop['holm_p']<.05:
        crop_text+='This is contrary evidence: this spatial probe favors 160 mm. It does not establish the best crop for a learned encoder.'
    else:
        crop_text+='It does not establish a reason to change the primary 130 mm crop.'
    small=rows['res_160x9_c130']
    small_text=(f"160² × 9 uses 5.67 GiB; its paired AUC difference is {small['delta_auc']:+.3f} "
                f"[{small['delta_ci95_lo']:+.3f}, {small['delta_ci95_hi']:+.3f}]. ")
    small_text+=('That interval fits inside the prespecified ±0.01 equivalence margin.'
                 if small['delta_ci95_lo']>-.01 and small['delta_ci95_hi']<.01 else
                 'This does not prove equivalence within ±0.01 AUC.')
    if small['delta_ci95_hi']<0 and small['holm_p']<.05:
        small_text+=" Allowing for the fact that we compared 14 alternatives still leaves evidence that 160² scored lower. The earlier claim that 160² is indistinguishable does not hold for this model."
    return base,resolution,slices,crop_text,small_text

def main():
    receipt=json.loads((EVIDENCE/'verifier_receipt.json').read_text())
    publication=json.loads(PUBLISHED.read_text())
    assert publication['status']=='ready' and isinstance(publication['is_private'],bool)
    assert publication['n_studies']==4407
    url=publication['url']
    assert url=='https://www.kaggle.com/datasets/dmitriigluzdov/rsna-knee-uint8-224-9-c130'
    base,resolution,slices,crop_text,small_text=conclusions(receipt)
    by_id={r['id']:r for r in receipt['rows']}
    three=by_id['slc_224x3_c130'];large=by_id['res_336x9_c130'];crop160=by_id['crp_224x9_c160']
    crop110=by_id['crp_224x9_c110']
    folds=(ROOT/'ops/assets/FOLDS_V1/folds.csv').read_bytes()
    payload=base64.b64encode(gzip.compress(folds)).decode()
    folds_code=f"import base64,gzip\nfrom pathlib import Path\n_ = Path('/kaggle/working/folds.csv').write_bytes(gzip.decompress(base64.b64decode({payload!r})))"
    summary=f"""# Knee MRI in 11 GiB

**TL;DR.** Keep all training studies, store fewer pixels. The primary download contains **all 4,407 training studies in 11.12 GiB of uint8 pixels**, without keeping roughly 500 GB of DICOM resident: six scan slots, nine slices per slot, 224 × 224 pixels, keeping a central region about 130 × 130 mm when cropping applies. These are training studies, not a count of unique people or the combined train and test sets.

- Independent image probe: **{base['auc']:.3f} held-out macro AUC** (95% interval {base['ci95_lo']:.3f}–{base['ci95_hi']:.3f}), on the same fixed 200 studies and five study folds.
- {resolution}
- {slices}
- **This is a storage and geometry result, not a medal.** It makes no leaderboard or foundation-model claim.

**[Download the primary cache]({url})** · Derived competition MRI; use is subject to the competition rules and MIRA terms.
"""
    measured=f"""## What we measured

We changed image resolution, slice count, or physical crop while keeping the study subset, labels, folds, and classifier fixed. A **slot** is one of six combinations of scan plane and the official `Fluid_Sensitive` flag. Nine slices means three groups of three neighboring images.

| What changes | What it means | What stays fixed |
| --- | --- | --- |
| Resolution: 128² to 336² | Resize the selected image to that many pixels per side using bilinear interpolation. More pixels give a finer grid, not a higher JPEG quality setting. | Nine slices; requested 130 mm crop |
| Slices: 3 to 15 per slot | Retain more or fewer images along the scan stack. | 224² pixels; requested 130 mm crop |
| Crop: 110 / 130 / 160 mm | **Side length of the central square we keep**, not the amount removed. Crop first, then resize. | 224² pixels; nine slices; identical storage |

All variants use the same eight-bit intensity conversion: clip the bottom/top 1% within each sampled, cropped series, then map to 0–255. We did not sweep bit depth or JPEG compression. Smaller resolution reduces the pixel count; smaller crop changes which anatomy those pixels cover.

The verifier reads **spatial edge maps and small image grids**, then fits a regularized linear classifier—a model whose weights are constrained to limit overfitting. It uses every retained slice at its cache resolution. This replaces the old model that saw only eight summary numbers per slot; it is a fixed image descriptor, not a pretrained or learned image encoder.

**Protocol:** 200 studies, exactly 40 from each locked `FOLDS_V1` fold; study-level five-fold validation; seed 2026. Targets are the 12 Pilkwang `report_labels_v2.csv` scores, binarized at 0.5. Every study is predicted by a model trained on the other folds. **Macro AUC** is the equal-weight average of the 12 target ranking scores; 0.5 is chance ranking and 1.0 is perfect ranking.

Error bars use 800 study resamples. Differences use the **same resampled studies** for both settings. These intervals do not include retraining or alternative-fold uncertainty. We call two settings practically equivalent only if their whole paired interval fits inside **±0.01 AUC**. A sanity check with randomly shuffled labels scored **{receipt['permutation_control_auc']:.3f}**. No settings were chosen using leaderboard scores.

Each of the 15 settings is now **built directly from sampled DICOM pixels**. The previous resolution sweep resized already converted eight-bit 336² images; this correction makes the 224² point match the downloadable cache. Within each sampled, cropped series, the lowest and highest 1% of intensities are clipped before scaling to one byte per pixel. Physical order uses image position along the slice stack, with `SliceLocation` and then `InstanceNumber` as fallbacks.
"""
    plots=f"""## Plots

**A** separates resolution from slice count, with crop fixed at 130 mm. Storage is the full 4,407-study pixel payload, not the 200-study probe size or a compressed archive download. The vertical line marks the primary cache. Gray triangles are the original three-slice extras with a narrower sampling window; they are not connected to the controlled families.

**B** compares the **width of the region kept**: 110, 130, or 160 mm, each resized to 224² × 9. Every point uses the same 11.12 GiB. Left means a tighter view; right means a wider view, not automatically better quality. Error bars show uncertainty in AUC, not image sharpness.
"""
    interpretation=f"""**How to read A: diminishing returns, not a proven plateau.** The resolution curve still rises: **{base['auc']:.3f} → {large['auc']:.3f} AUC**, while storage rises **11.12 → 25.02 GiB**. That is another **13.90 GiB** for an observed **{large['delta_auc']:+.3f} AUC**, with a paired 95% interval of **{large['delta_ci95_lo']:+.3f} to {large['delta_ci95_hi']:+.3f}**. The interval includes zero and a potentially useful gain: neither a reliable improvement nor strict equivalence is established. We keep 224² as a storage compromise. The existing 256² and 288² points already fill the gap; more resolution settings would not fix uncertainty from evaluating only 200 studies.

{slices} Three slices change AUC by **{three['delta_auc']:+.3f}**, with a paired 95% interval of **{three['delta_ci95_lo']:+.3f} to {three['delta_ci95_hi']:+.3f}**. We keep nine slices, but do not present the old “only real quality cliff” claim as established by this model.

{small_text} Only the primary 224² cache is published.

**How to read B: more context competes with detail.** Both a 110 mm and a 160 mm square end up as 224 × 224 pixels. When the requested crop applies, a 10 mm structure spans roughly **20 pixels at 110 mm**, versus **14 pixels at 160 mm**. The tighter view spends more pixels on central anatomy but discards more of the periphery. The wider view keeps more context but represents the same structure with fewer pixels. Neither must win. Resizing cannot create detail absent from the original scan.

**110 mm has the highest measured AUC, but is not a confirmed winner.** It scores **{crop110['auc']:.3f}**, versus **{base['auc']:.3f}** at 130 mm: difference **{crop110['delta_auc']:+.3f}**, paired 95% interval **{crop110['delta_ci95_lo']:+.3f} to {crop110['delta_ci95_hi']:+.3f}**. A tighter view may suit this edge-based verifier; sampling uncertainty may also explain the difference. This experiment does not identify the cause or establish the best crop for a trained image encoder. We retain the prespecified 130 mm compromise.

{crop_text} If the source image is too small for the requested square, the crop is skipped and the original field of view is resized instead. This happened at 160 mm for **{crop160.get('crop_skipped_fov',0)} of 977 available series** in the probe. Thus 160 mm is often effectively “no crop,” not a uniform 160 mm view. The millimetre-to-pixel example above applies only when cropping actually occurs.
"""
    download=f"""## What to download

**[dmitriigluzdov/rsna-knee-uint8-224-9-c130]({url})** — **4,407 studies, 11.12072 GiB pixel payload**, plus small headers and metadata. Archive transfer size may differ. The Dataset card includes a quick start, file and column descriptions, slot order, provenance and data-use terms.

The Dataset contains `pixels-000.npy` … `pixels-034.npy`, `studies.csv`, `slot_mask.npy`, `SPEC.json`, an audit, and licence notices. `SPEC.json` records the shape, geometry, exact byte counts, and SHA-256 checksums. **uint8** means one byte per pixel; **GiB** means 1,073,741,824 bytes. Each shard holds up to 128 studies. A zero slot with a zero mask means no matching series in the official metadata, not a healthy knee.

After attaching the Dataset with the required access, read one study without loading a whole shard into memory:

```python
from pathlib import Path
import numpy as np
import pandas as pd

# Use the mount path shown in your notebook's Input panel.
cache = Path('/kaggle/input/rsna-knee-uint8-224-9-c130')
index = pd.read_csv(cache / 'studies.csv')
i = 0                             # index row of the study you want
r = index.iloc[i]
study = np.load(cache / r['shard'], mmap_mode='r', allow_pickle=False)[int(r['row'])]
mask = np.load(cache / 'slot_mask.npy', allow_pickle=False)[i]
print(study.shape, study.dtype)    # (6, 9, 224, 224), uint8
```

Slot order: sagittal fluid, coronal fluid, axial fluid, sagittal structural, coronal structural, axial structural. The sampling window 0.35–0.65 refers to positions along the ordered slice stack, not an intensity window.

**Save & Run rebuilds the measurements and the full primary cache on CPU with internet off.** Rebuilt MRI is stored in `/kaggle/temp/rsna-knee-cache`, an ephemeral session directory, rather than saved notebook output. The linked Dataset holds the persistent download. Expand hidden code to inspect implementation.
"""
    caveats="""## Caveats and credit

This is a **lossy cache, not lossless DICOM**: it retains selected slices, crops/resizes them, and compresses intensities to eight bits. It cannot recover discarded anatomy or scanner metadata. The full corpus is cached; only 200 studies were used for the quality probe. Weak labels come from report processing and may be wrong. The six scan slots can be absent. Slice count and sampled anatomical coverage change together. A small fixed descriptor can miss benefits that a trained image encoder would find.

The requested crop uses row pixel spacing and is skipped for short fields of view or missing spacing; the audit counts those cases and anisotropic spacing. We preserve this existing geometry instead of silently changing it. No filename ordering or broken-slice substitution is allowed. These limitations prevent a claim that 11 GiB is universally optimal.

Derived MRI remains governed by the [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) and [RSNA MIRA licence](http://rsna.org/mira-license). The cache is intended for participants who accepted those terms, is not for non-participants, and does not replace obtaining the official training data. No reports, report lexicon, or `train.csv` are shipped.

Geometry credit: [Steven Lee's CPU pixel cache](https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache), Apache-2.0, reimplemented and modified here. No report or lexicon code was copied. Labels: [Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels). The code licence does not override MRI access terms.
"""
    code=(HERE/'pipeline.py').read_text(encoding='utf-8')
    # Figures are rendered in their own visible-output cell, after the hidden work.
    code=code.replace('    plot_curve(metrics,out)\n','    # Render separately in the public notebook.\n')
    run="""from IPython.utils.capture import capture_output
with capture_output() as build_log:
    receipt,spec=run_all('/kaggle/working/folds.csv', '/kaggle/working/evidence', '/kaggle/temp/rsna-knee-cache')
_ = Path('/kaggle/working/build.log').write_text(build_log.stdout,encoding='utf-8')
"""
    for filename in ['NOTICE.md','LICENSE-APACHE-2.0.txt']:
        run+=f"_ = Path('/kaggle/temp/rsna-knee-cache/{filename}').write_text({(HERE/filename).read_text(encoding='utf-8')!r},encoding='utf-8')\n"
    cells=[cell('markdown',summary,'tldr'),cell('markdown',measured,'measured'),
           cell('code',folds_code,'folds'),cell('code',code,'implementation'),cell('code',run,'rebuild'),
           cell('markdown',plots,'plots'),
           cell('code',"plot_curve(pd.read_csv('/kaggle/working/evidence/verifier_metrics.csv'),Path('/kaggle/working/evidence'))",'figures'),
           cell('markdown',interpretation,'interpretation'),cell('markdown',download,'download'),
           cell('markdown',caveats,'caveats')]
    # Kaggle Quick Save does not render supplied code outputs. Markdown attachments
    # preserve the verified figures; Save & Run also regenerates both PNG files.
    figure_cell=next(c for c in cells if c['id']=='figures')
    figure_cell['metadata']['_kg_hide-output']=True
    saved_figures=cell('markdown',
        '![A: Quality versus storage](attachment:figure_A.png)\n\n'
        '![B: Crop comparison at fixed storage](attachment:figure_B.png)\n\n'
        'Figures from the completed CPU verifier run. Save & Run regenerates both PNGs.',
        'saved-figures')
    saved_figures['attachments']={}
    for name in ['figure_A.png','figure_B.png']:
        saved_figures['attachments'][name]={
            'image/png':base64.b64encode((EVIDENCE/name).read_bytes()).decode()}
    cells.insert(cells.index(figure_cell)+1,saved_figures)
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'cache_verifier_receipt_sha256':hashlib.sha256((EVIDENCE/'verifier_receipt.json').read_bytes()).hexdigest()})
    PUBLIC.mkdir(exist_ok=True)
    path=PUBLIC/'rsna-knee-on-a-storage-budget.ipynb'
    path.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    metadata=dict(id='dmitriigluzdov/knee-mri-in-11-gib',id_no=133521917,title='Knee MRI in 11 GiB',
        code_file=path.name,language='python',kernel_type='notebook',is_private=False,
        enable_gpu=False,enable_tpu=False,enable_internet=False,
        dataset_sources=['pilkwang/rsna-knee-llm-labels'],competition_sources=['rsna-knee-abnormality-detection'],
        kernel_sources=[],model_sources=[])
    (PUBLIC/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for c in cells:
        if c['cell_type']=='code':
            compile(c['source'],f"cell:{c['id']}",'exec')
            assert c['metadata']['jupyter']['source_hidden'] is True
    assert sum(len(c.get('attachments',{})) for c in cells)==2
    print('Built',path,'with hidden runnable code and two measured figures')

if __name__=='__main__':main()
