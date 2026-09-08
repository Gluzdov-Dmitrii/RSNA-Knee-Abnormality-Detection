"""Prepare data-card metadata while preserving visibility; does not upload data."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/cache_budget_recheck/metadata_update'

def main():
    previous=json.loads((OUT.parent/'metadata_before/dataset-metadata.json').read_text())
    meta=previous.get('info',previous)
    spec=json.loads((OUT.parent/'cache/SPEC.json').read_text())
    meta.update(subtitle='4,407 studies in 11.12 GiB: 6 slots, 9 slices, NumPy uint8; participant use',
        description=(HERE/'DATA_CARD.md').read_text(encoding='utf-8'),
        isPrivate=meta.get('isPrivate',True),keywords=['computer vision','image','health conditions','multilabel classification','healthcare'],
        expectedUpdateFrequency='never',
        userSpecifiedSources='https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data\n'
          'https://www.kaggle.com/code/stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache\n'
          'https://www.kaggle.com/code/dmitriigluzdov/knee-mri-in-11-gib')
    descriptions={
        'studies.csv':'Study-to-shard index for all 4,407 training studies. The CSV row index also indexes slot_mask.npy. Join authorized labels by StudyInstanceUID, not row order.',
        'slot_mask.npy':'uint8 array (4407,6), aligned with studies.csv rows and the documented slot order. 1 means a selected series exists; 0 means an absent slot filled with zeros. Not a disease label.',
        'SPEC.json':'Machine-readable cache specification: image size, slice sampling, physical crop, slot order, dtype, study count, and SHA-256 checksums for every shard and index/mask/audit. Private wording records the initial build, not current visibility.',
        'AUDIT.json':'Full 4,407-study conversion counters: 21,334 physically ordered series, 190,686 decoded slices, 5,108 absent slots, 20,976 crops applied and 358 crops skipped for short fields of view.',
        'NOTICE.md':'Geometry attribution and MRI data-use restrictions supplied with the original private build. Current hosting visibility is on the Dataset page; competition and MIRA terms remain applicable.',
        'LICENSE-APACHE-2.0.txt':'Apache License 2.0 for credited geometry code. This does not relicense the competition MRI pixels; see the data card and NOTICE.md.'}
    for s in spec['shards']:
        descriptions[s['file']]=(f"Pixel shard with {s['n_studies']} studies; uint8 shape ({s['n_studies']},6,9,224,224). "
            'Axes: study, slot, retained slice, height, width. Find IDs/rows in studies.csv and missing slots in slot_mask.npy. '
            "Read with np.load(path,mmap_mode='r',allow_pickle=False)[row]. SHA-256 is recorded in SPEC.json.")
    meta['data']=[dict(name=k,description=v,totalBytes=(OUT.parent/'cache'/k).stat().st_size) for k,v in descriptions.items()]
    next(d for d in meta['data'] if d['name']=='studies.csv')['columns']=[
        dict(name='StudyInstanceUID',description='Unique official training-study identifier. Join key to authorized labels; not a unique-person identifier.'),
        dict(name='shard',description='NumPy filename containing this study, pixels-000.npy through pixels-034.npy.'),
        dict(name='row',description='Zero-based study position inside the named shard; use np.load(shard,mmap_mode="r")[row].')]
    assert len(meta['data'])==41 and len(meta['keywords'])==5
    OUT.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig=plt.figure(figsize=(5.6,2.8),dpi=100,facecolor='#102333')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,560),ylim=(0,280));ax.axis('off')
    for side in (32,454):
        for j in range(6):
            ax.add_patch(Rectangle((side+j*2,70+j*13),52,64,facecolor='#153e50',edgecolor='#3f94a8',lw=1))
    ax.text(280,221,'RSNA KNEE MRI',ha='center',color='white',fontsize=19,weight='bold')
    ax.text(280,155,'11.12 GiB',ha='center',color='#77d9c0',fontsize=35,weight='bold')
    ax.text(280,114,'4,407 training studies',ha='center',color='white',fontsize=14)
    ax.text(280,78,'224 × 224 · 9 slices · 6 slots',ha='center',color='#b9d1de',fontsize=12)
    ax.text(280,39,'NUMPY UINT8 PIXEL CACHE',ha='center',color='#b9d1de',fontsize=10)
    fig.savefig(OUT/'cover.png',dpi=100,facecolor=fig.get_facecolor());plt.close(fig)
    meta['image']='cover.png'
    (OUT/'dataset-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print('Prepared 41 file descriptions, 3 column descriptions, 5 tags, provenance and update frequency.')

if __name__=='__main__':main()
