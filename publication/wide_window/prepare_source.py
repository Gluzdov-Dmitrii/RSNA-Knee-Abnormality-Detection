"""Freeze an isolated P04 source snapshot without changing deployed P03."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'publication/consensus_coatnet'
DEST=ROOT/'artifacts/wide_window/source'
INIT_SHA='a7d32ac8c3f006d2b60d1837a90a27c1cf6ed7ad80bdef018c8de52decc4a8c3'

def build():
    DEST.mkdir(parents=True,exist_ok=True)
    provenance={}
    for name in ['common.py','model.py','train.py','geometry.py','infer.py','kaggle_qa.py',
                 'precision_reference.py','audit_cache_hashes.py']:
        original=(BASE/name).read_text(encoding='utf-8')
        source=original
        if name=='common.py':
            old='GEOMETRY = dict(img=224,n_slices=9,crop_mm=130,window=[.35,.65])'
            assert source.count(old)==1
            source=source.replace(old,'GEOMETRY = dict(img=224,n_slices=9,crop_mm=130,window=[.10,.90])')
        if name=='train.py':
            old="if init['geometry']!=GEOMETRY or init['targets']!=TARGETS or init['arch']!=ARCH: raise ValueError('Initialization contract differs')"
            new=f"""if sha256(a.initialization)!={INIT_SHA!r}: raise ValueError('Must reuse exact generic P01 initialization')
    expected_init_geometry=dict(img=224,n_slices=9,crop_mm=130,window=[.35,.65])
    if init['geometry']!=expected_init_geometry or init['targets']!=TARGETS or init['arch']!=ARCH:
        raise ValueError('Initialization contract differs')
    # Slice selection is not a model tensor; keep the exact initial tensors,
    # while recording the new data geometry in config/checkpoints.
    baseline_root=Path(a.cache).parent/'rsna-knee-uint8-224-9-c130'
    baseline_index=pd.read_csv(baseline_root/'studies.csv',dtype={{UID:str}})
    if cache.index[UID].tolist()!=baseline_index[UID].tolist():raise ValueError('P01 cache row order differs')
    baseline_mask=np.load(baseline_root/'slot_mask.npy',mmap_mode='r',allow_pickle=False)
    if not np.array_equal(cache.mask,baseline_mask):raise ValueError('P01 slot masks differ')"""
            assert source.count(old)==1
            source=source.replace(old,new)
            source=source.replace("initialization_sha256=sha256(a.initialization),folds_sha256=sha256(a.folds),",
                "initialization_sha256=sha256(a.initialization),initialization_geometry=expected_init_geometry,folds_sha256=sha256(a.folds),")
        compile(source,name,'exec')
        (DEST/name).write_text(source,encoding='utf-8')
        provenance[name]=dict(base_sha256=hashlib.sha256(original.encode()).hexdigest(),
            p04_sha256=hashlib.sha256(source.encode()).hexdigest(),changed=source!=original)
    for name in ['NOTICE.md','LICENSE-APACHE-2.0.txt']:
        (DEST/name).write_bytes((BASE/name).read_bytes())
    remote=(BASE/'remote_job.py').read_text(encoding='utf-8')
    remote=remote.replace("RUN=PROJECT/'runs/20260910T0913Z-codex-consensus'",
        "RUN=PROJECT/'runs/20260912-codex-wide-window'\nBASE_RUN=PROJECT/'runs/20260910T0913Z-codex-consensus'")
    remote=remote.replace("choices=['smoke','smoke2','pair','verified_pair','full']","choices=['smoke','pilot','full']")
    remote=remote.replace("arms=['pilkwang'] if a.mode.startswith('smoke') or a.mode=='full' else ['smoke','pilkwang','median']","arms=['pilkwang']")
    remote=remote.replace("PROJECT/'data/rsna-knee-uint8-224-9-c130'","PROJECT/'data/rsna-knee-uint8-224-9-c130-w10-90'")
    remote=remote.replace("RUN/'inputs/gold_uids.csv'","BASE_RUN/'inputs/gold_uids.csv'")
    remote=remote.replace("RUN/'prepared_correctnorm/initialization.pt'","BASE_RUN/'prepared_correctnorm/initialization.pt'")
    compile(remote,'remote_job.py','exec')
    (DEST/'remote_job.py').write_text(remote,encoding='utf-8')
    (DEST/'remote_preflight.py').write_bytes(Path(__file__).with_name('remote_preflight.py').read_bytes())
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob('*.py')}
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (DEST/'SOURCE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':build()
