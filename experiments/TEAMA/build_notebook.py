"""Build the TEAMA notebook (variant A) from pinned public sources.

Base: laimerbotrolenk/rsna-four-model-ensemble V1 (public 0.957): Raptor-384 TTA, Raptor-224 plain+mirror,
R7 compact ConvNeXt, S6 all-data ConvNeXt.  Heliosli's R6 compact ConvNeXt (bundle fe055406) with flip TTA, exactly as
in our team's 079 (0.957), replaces R7: the 16.10 rehearsal measured R6-R7 rank correlation 0.968 and ~9.0 h for all
five members on ~1300 studies, so only one of the two near-duplicate Heliosli members fits.  R7 stays buildable
(WITH_R7) for reference.  Weights follow a rule written before any run (see RULE below).

    python build_notebook.py --rehearsal 0    -> kaggle/scored/   (submission build)
    python build_notebook.py --rehearsal 64   -> kaggle/rehearsal/ (timing run on 64 seeded TRAIN studies)
The two builds differ only in REHEARSAL_N; rehearsal-only code is guarded by `if REHEARSAL_N:`.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAIMER = HERE / 'source' / 'laimer_v1.ipynb'
LAIMER_SHA = '50ded2186d86e172e85a8928e8b93b3cdd0c9112117d2267ac059e651793a003'
K079 = HERE / 'source' / 'k079_v.ipynb'
K079_SHA = '82a52f2bcbd2648961b54ce9675163a78c69468cd499a222218ecf752d6a4e88'

WITH_R7 = False
VARIANT = 'A'      # set by --variant; B = A with R6 plain (no flip TTA) + hengck23 MaViT-288 as a third family

RULE_B = """\
**Weight rule B (written 2026-10-16, before any run of B; not fitted on labels or on the leaderboard).**
Variant A's rule, two changes. (1) R6 runs without flip TTA: TTA doubles its GPU time (~4.2 h vs ~2.2 h for ~1300
studies on the rehearsal), and the saved time buys a third model family. (2) hengck23's MaViT-288 (public 0.935; a
linear-attention ViT, its own checkpoint and labels, no dataset shared with the 0.95+ public blends) gets 0.15 of the
final rank; Raptor and compact families keep 0.425 each (inside them as in A). 0.15 is deliberately small: the member
is the weakest standalone, and the point is an independent error pattern, not a larger share.
Failures: as in A; MaViT's all-zero rows (its own failure convention) count as failed studies."""

RULE = """\
**Weight rule (written 2026-10-16, before any scored run; not fitted on labels or on the leaderboard).**
Two families, equal weight: Raptor CoAtNet family 0.5, compact ConvNeXt family 0.5.
Inside Raptor: 384 = 0.65, 224 = 0.35 (the split every public Raptor blend uses).
Inside compact: equal shares for every independent member — Heliosli R6 with flip TTA 1/2, nartaa S6 all-data 1/2.
Heliosli's R7 is left out: on the 64-study rehearsal it ranked studies like R6 (Spearman 0.968) and the five-member
notebook needs ~9 h for the ~1300 test studies, over the run limit.
Per study, a failed member is dropped and the remaining weights inside its family are renormalised;
any member failing on more than 3 % of the studies stops the notebook (no silent degraded submission)."""


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def src(cell):
    return ''.join(cell['source'])


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text}


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}


def must_replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


SETTINGS = """\
# Frozen recipe settings.  The scored build has REHEARSAL_N = 0; the rehearsal build differs only in this number.
REHEARSAL_N = {rehearsal}        # >0: run every stage on N seeded TRAIN studies instead of test (timing / sanity only)
REQUIRE_S6 = True
TOTAL_HOURS = 8.3
S6_RESERVE_MINUTES = 110         # S6's original setup/retry allowance (from the four-model source)
R6_RESERVE_MINUTES = {r6_reserve}        # time kept for the R6 stage when R7 runs
MAX_FAILURE_FRACTION = 0.03
RECIPE = 'reference_aleksei'     # only used by the inherited source; the five-model blend below writes submission.csv
"""

REHEARSAL_ROOT = """\
# Rehearsal only: a fake competition root whose "test" studies are N seeded TRAIN studies (symlinked DICOM folders).
FAKE_KAGGLE = Path('/tmp/a5_fake')
FAKE_COMP = FAKE_KAGGLE / 'input' / 'competitions' / 'rsna-knee-abnormality-detection'
if REHEARSAL_N:
    import numpy as np, pandas as pd
    _comp = next(p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                             Path('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'train.csv').is_file())
    _tr = pd.read_csv(_comp / 'train.csv', dtype={'StudyInstanceUID': str})
    _gold = sorted(_tr.loc[_tr['ACL'].notna(), 'StudyInstanceUID'])[:REHEARSAL_N]
    _rest = sorted(set(_tr['StudyInstanceUID']) - set(_gold))
    _ids = sorted(_gold + np.random.default_rng(2026).choice(_rest, REHEARSAL_N - len(_gold), replace=False).tolist())
    if FAKE_KAGGLE.exists():
        shutil.rmtree(FAKE_KAGGLE)
    (FAKE_COMP / 'test_series').mkdir(parents=True)
    pd.DataFrame({'StudyInstanceUID': _ids}).to_csv(FAKE_COMP / 'test.csv', index=False)
    _ser = pd.read_csv(_comp / 'train_series.csv', dtype=str)
    _ser[_ser.StudyInstanceUID.isin(_ids)].to_csv(FAKE_COMP / 'test_series.csv', index=False)
    _ss = pd.DataFrame({'StudyInstanceUID': _ids})
    for _c in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']:
        _ss[_c] = 0.5
    _ss.to_csv(FAKE_COMP / 'sample_submission.csv', index=False)
    for _u in _ids:
        os.symlink(_comp / 'train_series' / _u, FAKE_COMP / 'test_series' / _u)
    for _p in Path('/kaggle/input').iterdir():          # every dataset mount, so S6's own mount search still works
        if _p.name != 'competitions' and _p.resolve() != _comp.resolve():
            os.symlink(_p, FAKE_KAGGLE / 'input' / _p.name)
    os.symlink('/kaggle/working', FAKE_KAGGLE / 'working')
    os.environ['RSNA_FAKE_ROOT'] = str(FAKE_COMP)
    print('REHEARSAL', REHEARSAL_N, 'train studies;', len(_ser[_ser.StudyInstanceUID.isin(_ids)]), 'series', flush=True)
"""

PATCH_RAPTOR = """\
# Rehearsal only: point the unchanged Raptor script at the fake root.  The scored build leaves the file untouched.
if REHEARSAL_N:
    _p = Path('{path}')
    _t = _p.read_text()
    assert _t.count('def find_test_root():\\n') == 1
    _p.write_text(_t.replace('def find_test_root():\\n',
        "def find_test_root():\\n    if os.environ.get('RSNA_FAKE_ROOT'):\\n        return os.environ['RSNA_FAKE_ROOT']\\n"))
print('{path}', sha256('{path}'), flush=True)
"""

FUSION = (HERE / 'fusion_cell.py').read_text(encoding='utf-8')

REHEARSAL_REPORT = r'''# Rehearsal only: per-stage seconds per study and the extrapolation to the ~1300 hidden test studies.
if REHEARSAL_N:
    sec = receipt['seconds']
    per = {k: v / REHEARSAL_N for k, v in sec.items() if k != 'total'}
    est = {k: round(v * 1300 / 3600, 2) for k, v in per.items()}
    report = {'n': REHEARSAL_N, 'seconds': sec, 'seconds_per_study': {k: round(v, 2) for k, v in per.items()},
              'hours_for_1300_upper_bound': est, 'hours_total_for_1300': round(sum(est.values()), 2),
              'note': 'stage seconds include fixed setup/compile cost, so the extrapolation is an upper bound'}
    try:
        lab = pd.read_csv(_comp / 'train.csv', dtype={UID: str}).set_index(UID).loc[test_ids, LABELS]
        report['gold_rows_in_sample'] = int(lab.notna().all(axis=1).sum())
        from sklearn.metrics import roc_auc_score
        g = lab.notna().all(axis=1).to_numpy()
        def _auc(x):
            return round(float(np.mean([roc_auc_score(lab.to_numpy()[g, j], x[g, j]) for j in range(12)])), 4)
        # diagnostic only: most public members were trained or selected with these 58 studies (contaminated)
        report['gold58_mean_auc'] = {k: _auc(v) for k, v in P.items()}
        report['gold58_mean_auc']['blend'] = _auc(final)
    except Exception as e:
        report['gold_rows_in_sample'] = repr(e)[:200]
    (WORK / 'a5_rehearsal_report.json').write_text(json.dumps(report, indent=1))
    print('REHEARSAL_REPORT', json.dumps(report), flush=True)
'''


MAVIT_CELLS = (
    "MAVIT_OK = False\nMAVIT_SECONDS = None\n"
    "_t = time.time()\n"
    "run_stage([sys.executable, '-u', '/tmp/mavit_run.py'], 'b_mavit', 90 * 60)\n"
    "MAVIT_SECONDS = time.time() - _t\n"
    "assert (WORK / 'mavit_sub.csv').is_file() and (WORK / 'mavit_status.csv').is_file()\n"
    "MAVIT_OK = True\n"
    "print('MAVIT_STAGE_DONE', round(MAVIT_SECONDS, 1), flush=True)\n")


def build(rehearsal, r6_reserve):
    assert sha(LAIMER) == LAIMER_SHA and sha(K079) == K079_SHA
    L = json.load(open(LAIMER, encoding='utf-8'))['cells']
    K = json.load(open(K079, encoding='utf-8'))['cells']
    cells = []
    cells.append(md('# RSNA Knee — variant ' + VARIANT + ': public model families, fixed family weights\n\n'
                    'Private team notebook (Kneerds).  Rank blend of every public model family we can run: '
                    "nartaa's Raptor CoAtNet-384 (mirror TTA) and 224-crop (plain + mirror), Heliosli's compact ConvNeXt "
                    "R6 (bundle fe055406, flip TTA)" + (" and R7 (bundle 8c7b666e)" if WITH_R7 else "") + ", and nartaa's all-data S6 ConvNeXt.  "
                    'All model code and weights are the published ones, run unchanged in their own stages.  The goal is a '
                    'stable private-LB ensemble, not public-LB tuning.\n\n' + RULE
                    + ('\n\n' + RULE_B if VARIANT == 'B' else '')))
    cells.append(md(src(L[2])))                                   # required credits + OAI acknowledgement
    cells.append(md('## Additional credits\n\nThe 224-pixel stage and its share follow '
                    "varshithakolupuri's notebook; the four-model frame follows AlekseiProvorov's and laimerbotrolenk's "
                    "public notebooks; R6 with flip TTA follows Heliosli's public notebooks."
                    + (" MaViT-288 stage: hengck23's public notebook and checkpoint (dataset hengck23/hengck23-rnsa-knee); "
                       "its DICOM reader is modified from nartaa's; MaViT encoder from https://github.com/qhfan/MALA "
                       "(Fan, Rectifying Magnitude Neglect in Linear Attention, ICCV 2025)." if VARIANT == 'B' else '')))
    cells.append(code(SETTINGS.format(rehearsal=rehearsal, r6_reserve=r6_reserve)))
    setup = src(L[4])
    setup = must_replace(setup, "assert RECIPE in ('mirror_mix_half', 'reference_aleksei')\n", '')
    setup = must_replace(setup, "'raptor224_status.csv', 'ours_sub.csv',", "'raptor224_status.csv', 'ours_sub.csv', 'r6_sub.csv', 'r7_sub.csv',")
    cells.append(code(setup))
    cells.append(code(REHEARSAL_ROOT))
    cells.append(code(src(L[5])))
    cells.append(code(src(L[6])))                                 # v33_io / v33_fusion helpers (unchanged)
    cells.append(md('## Raptor-384 (published script, unchanged)'))
    cells.append(code(src(L[8])))
    cells.append(code(PATCH_RAPTOR.format(path='/tmp/raptor_run.py')))
    cells.append(code(src(L[9])))
    cells.append(md('## Raptor-224, plain and mirror (published script as in the four-model source)'))
    cells.append(code(src(L[11])))
    cells.append(code(PATCH_RAPTOR.format(path='/tmp/raptor224_run.py')))
    cells.append(code(src(L[12])))
    if VARIANT == 'B':
        cells.append(md('## MaViT-288 (hengck23, published inference loop; third family)'))
        cells.append(code('%%writefile /tmp/mavit_run.py\n' + (HERE / 'mavit_stage.py').read_text(encoding='utf-8')))
        cells.append(code(MAVIT_CELLS))

    # R7 cells are derived (the R6 cells are built from them) but only appended when R7 is a member.
    r7_setup = src(L[14])
    r7_setup = must_replace(r7_setup, "BUNDLE = TMP / 'bundle'\n", "BUNDLE = TMP / ('bundle_' + BUNDLE_ID)\n")
    if WITH_R7:
        cells.append(md('## R7 compact ConvNeXt (bundle 8c7b666e, published reader, no flip TTA)'))
        cells.append(code(r7_setup))
        cells.append(code("if REHEARSAL_N:\n    DATA = FAKE_COMP\nprint('DATA', DATA, flush=True)\n"))
    r7_run = src(L[15])
    r7_run = must_replace(r7_run, "    deadline -= (S6_RESERVE_MINUTES + 10) * 60\n",
                          "    deadline -= (S6_RESERVE_MINUTES + 10) * 60\ndeadline -= R6_RESERVE_MINUTES * 60\n")
    r7_run = must_replace(r7_run, "pred = TMP / 'prediction_v33'", "pred = TMP / 'prediction_r7'")
    r7_run = must_replace(r7_run, "run_stage(cmd, 'v33_r7', deadline - time.time(), env=env)",
                          "_t7 = time.time()\nrun_stage(cmd, 'a5_r7', deadline - time.time(), env=env)\nR7_SECONDS = time.time() - _t7")
    r7_run = must_replace(r7_run, "WORK / (log.parent.name + '_log.txt')", "WORK / ('r7_' + log.parent.name + '_log.txt')")
    assert r7_run.count('ours_sub.csv') == 4
    r7_run = r7_run.replace('ours_sub.csv', 'r7_sub.csv')
    r7_run = must_replace(r7_run, "(WORK / 'submission_receipt.json')", "(WORK / 'r7_submission_receipt.json')")
    r7_run += "\nPRED_R7 = pred\n"
    if WITH_R7:
        cells.append(code(r7_run))

    # R6: the same published reader with the R6 bundle and flip TTA, exactly the switches of our 079 (0.957).
    cells.append(md('## R6 compact ConvNeXt (bundle fe055406, ' + ('flip TTA as in team submission 079)' if VARIANT == 'A' else 'plain, no flip TTA)')))
    r6_setup = r7_setup
    r6_setup = must_replace(r6_setup, "BUNDLE_ID = '8c7b666e27dc0e8f'", "BUNDLE_ID = 'fe055406fad6a12f'")
    r6_setup = must_replace(r6_setup, "C_SLUG = 'rsna-knee-compact-r7cnxfsss'   # R7 bundle (single-version dataset; manifest identical to r6-frozen v2)",
                            "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'   # Heliosli's single-version R6 bundle (manifest asserted below)")
    if VARIANT == 'A':
        r6_setup = must_replace(r6_setup, "FLIP_TTA = False   # [d957] package F: no flip TTA", "FLIP_TTA = True   # [d957] package E: v4c's flip TTA on R6")
    else:
        r6_setup = must_replace(r6_setup, "FLIP_TTA = False   # [d957] package F: no flip TTA", "FLIP_TTA = False   # variant B: R6 plain (TTA time goes to MaViT)")
    assert r6_setup.count("'bundle_' + BUNDLE_ID") == 1
    # Kaggle rejects notebooks over ~1 MB: reuse the R7 cell's SOURCES / CODE_SHA / SOURCE_SHA (the reader sources are
    # byte-identical for R6 and R7; every file is still re-checked against SOURCE_SHA when this cell writes it).
    if WITH_R7:
        _lines = r6_setup.split('\n')
        _drop = [l for l in _lines if l.startswith(('SOURCES = ', 'CODE_SHA = ', 'SOURCE_SHA = '))]
        assert len(_drop) == 3, len(_drop)
        r6_setup = '\n'.join(['# SOURCES, CODE_SHA and SOURCE_SHA: defined by the R7 cell above (identical reader sources).']
                              + [l for l in _lines if l not in _drop])
    cells.append(code(r6_setup))
    cells.append(code("if REHEARSAL_N:\n    DATA = FAKE_COMP\nassert ('--flip-tta' in WORKER_ARGS) == FLIP_TTA\nprint('DATA', DATA, 'flip', FLIP_TTA, flush=True)\n"))
    r6_run = r7_run
    r6_run = must_replace(r6_run, "deadline -= R6_RESERVE_MINUTES * 60\n", "")
    r6_run = must_replace(r6_run, "'Insufficient R7 budget while reserving the S6 stage'", "'Insufficient R6 budget while reserving the S6 stage'")
    r6_run = must_replace(r6_run, "pred = TMP / 'prediction_r7'", "pred = TMP / 'prediction_r6'")
    r6_run = must_replace(r6_run, "_t7 = time.time()\nrun_stage(cmd, 'a5_r7', deadline - time.time(), env=env)\nR7_SECONDS = time.time() - _t7",
                          "_t6 = time.time()\nrun_stage(cmd, 'a5_r6', deadline - time.time(), env=env)\nR6_SECONDS = time.time() - _t6")
    r6_run = r6_run.replace("'r7_", "'r6_").replace("r7_sub.csv", "r6_sub.csv").replace("r7_submission_receipt", "r6_submission_receipt")
    r6_run = must_replace(r6_run, "\nPRED_R7 = pred\n", "\nPRED_R6 = pred\n")
    assert 'r7' not in r6_run.lower().replace('pred_r7', ''), 'R7 name left in the R6 cell'
    cells.append(code(r6_run))

    cells.append(md('## S6 all-data ConvNeXt (published script, unchanged)'))
    s6 = src(L[17])
    s6 = must_replace(s6, "K1_KAGGLE_ROOT='/kaggle'", "K1_KAGGLE_ROOT=str(FAKE_KAGGLE) if REHEARSAL_N else '/kaggle'")
    cells.append(code(s6))
    cells.append(md('## Blend\n\n' + RULE))
    cells.append(code(FUSION))
    cells.append(code(REHEARSAL_REPORT))
    nb = {'cells': cells, 'metadata': json.load(open(LAIMER, encoding='utf-8'))['metadata'], 'nbformat': 4, 'nbformat_minor': 4}
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rehearsal', type=int, default=0)
    ap.add_argument('--r6-reserve', type=int, default=150)
    ap.add_argument('--variant', choices=('A', 'B'), default='A')
    a = ap.parse_args()
    global VARIANT
    VARIANT = a.variant
    name = ('rehearsal' if a.rehearsal else 'scored') + ('' if VARIANT == 'A' else '_b')
    out = HERE / 'kaggle' / name
    out.mkdir(parents=True, exist_ok=True)
    slug = 'rsna-team-' + VARIANT.lower() + '-ensemble' + ('-rehearsal' if a.rehearsal else '')
    nb = build(a.rehearsal, a.r6_reserve)
    path = out / (slug + '.ipynb')
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    meta = {'id': 'dmitriigluzdov/' + slug, 'title': slug, 'code_file': slug + '.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'dataset_sources': ['nartaa/rsna-knee-publication-swa-weights-20261007']
                               + (['heliosli/rsna-knee-compact-r7cnxfsss'] if WITH_R7 else [])
                               + ['heliosli/rsna-knee-r6-frozen-fe055406-w1', 'nartaa/rsna-knee-s6fall-weights',
                                  'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']
                               + (['hengck23/hengck23-rnsa-knee'] if VARIANT == 'B' else []),
            'competition_sources': ['rsna-knee-abnormality-detection'], 'kernel_sources': [], 'model_sources': [],
            'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
            'machine_shape': 'NvidiaTeslaT4'}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    print(path, sha(path), len(nb['cells']), 'cells')


if __name__ == '__main__':
    main()
