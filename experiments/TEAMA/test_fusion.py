"""CPU test of the generated variant-A fusion cell on synthetic member outputs.

Checks: weights reproduce a hand-computed blend (scored member set, and the optional R7 set); a failed study of one
member is dropped inside its family; the 3 % gate stops the notebook; the agreement gate stops on an unrelated member.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NB = HERE / 'kaggle' / ('scored_b/rsna-team-b-ensemble.ipynb' if 'B' in sys.argv else 'scored/rsna-team-a-ensemble.ipynb')
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
          'Synovitis', "Baker's", 'Contusion', 'Fracture']
cells = [''.join(c['source']) for c in json.load(open(NB, encoding='utf-8'))['cells']]
helpers = next(c for c in cells if c.startswith('(V33 / "v33_io.py")'))
fusion = next(c for c in cells if c.startswith('# Variant A fusion'))
assert not any("'a5_r7'" in c or '8c7b666e27dc0e8f' in c for c in cells), 'R7 stage in scored build'


def write(path, ids, x):
    df = pd.DataFrame(x, columns=LABELS)
    df.insert(0, 'StudyInstanceUID', ids)
    df.to_csv(path, index=False)


def scenario(n=200, seed=0, fail_r6=(), unrelated_s6=False, with_r7=False, mavit=False, mavit_fail=()):
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp())
    work, data, v33 = tmp / 'work', tmp / 'data', tmp / 'v33'
    for d in (work, data, v33):
        d.mkdir()
    ids = ['1.2.%d' % i for i in range(n)]
    pd.DataFrame({'StudyInstanceUID': ids}).to_csv(data / 'test.csv', index=False)
    z = rng.normal(size=(n, 12))
    mk = lambda s: 1 / (1 + np.exp(-(z + s * rng.normal(size=(n, 12)))))
    p384, plain, mirror, p7, p6r, s6 = (mk(0.5) for _ in range(6))
    if unrelated_s6:
        s6 = rng.uniform(size=(n, 12))
    tta = 0.5 * (plain + mirror)
    write(work / 'raptor_raw.csv', ids, p384)
    write(work / 'raptor224_plain.csv', ids, plain)
    write(work / 'raptor224_mirror.csv', ids, mirror)
    write(work / 'raptor224_raw.csv', ids, tta)
    pd.DataFrame({'StudyInstanceUID': ids, 'plain_ok': 1, 'mirror_ok': 1, 'valid_slices': 10}).to_csv(
        work / 'raptor224_status.csv', index=False)
    write(work / 'r7_sub.csv', ids, p7)
    write(work / 'r6_sub.csv', ids, p6r)
    fail = set(fail_r6) if not isinstance(fail_r6, float) else set(range(int(fail_r6 * n)))
    for name, p, failed in (('r7', p7, set()), ('r6', p6r, fail)):
        d = tmp / ('pred_' + name) / 'worker0'
        d.mkdir(parents=True)
        keep = [i for i in range(n) if i not in failed]
        write(d / 'predictions.csv', [ids[i] for i in keep], p[keep])
    s6p = work / 's6.csv'
    write(s6p, ids, s6)
    pm = mk(0.8)
    pm[list(mavit_fail)] = 0.0
    write(work / 'mavit_sub.csv', ids, pm)
    pd.DataFrame({'StudyInstanceUID': ids, 'ok': [0 if i in mavit_fail else 1 for i in range(n)]}).to_csv(work / 'mavit_status.csv', index=False)
    g = {'WORK': work, 'DATA': data, 'V33': v33, 'Path': Path, 'json': json, 'time': __import__('time'),
         'NB_T0': 0.0, 'RAPTOR_SECONDS': 1.0, 'RAPTOR224_SECONDS': 1.0, 'R7_SECONDS': 1.0, 'R6_SECONDS': 1.0,
         'S6_SECONDS': 1.0, 'S6_OK': True, 'S6_PROBS': s6p, 'PRED_R6': tmp / 'pred_r6',
         'MAX_FAILURE_FRACTION': 0.03, 'REHEARSAL_N': 0, 'display': lambda *a: None}
    if with_r7:
        g['PRED_R7'] = tmp / 'pred_r7'
    if mavit:
        g['MAVIT_OK'], g['MAVIT_SECONDS'] = True, 1.0
    sys.path.insert(0, str(v33))
    exec(helpers, g)
    for m in ('v33_io', 'v33_fusion'):
        sys.modules.pop(m, None)
    exec(fusion, g)
    sys.path.remove(str(v33))
    out = pd.read_csv(work / 'submission.csv').set_index('StudyInstanceUID').loc[ids, LABELS].to_numpy()
    return out, dict(p384=p384, p224=tta, p7=p7, p6r=p6r, s6=s6, pm=pm, n=n)


def pct(x):
    return pd.DataFrame(x).rank(method='average', pct=True).to_numpy()


def test_weights():
    out, m = scenario()
    want = 0.5 * (0.65 * pct(m['p384']) + 0.35 * pct(m['p224'])) + 0.5 * (pct(m['p6r']) + pct(m['s6'])) / 2
    assert np.abs(out - want).max() < 1e-6, np.abs(out - want).max()


def test_weights_with_r7():
    out, m = scenario(with_r7=True)
    want = 0.5 * (0.65 * pct(m['p384']) + 0.35 * pct(m['p224'])) + 0.5 * (pct(m['p6r']) + pct(m['p7']) + pct(m['s6'])) / 3
    assert np.abs(out - want).max() < 1e-6, np.abs(out - want).max()


def test_member_failure_dropped():
    out, m = scenario(fail_r6=(5,))
    ok = np.arange(m['n']) != 5
    r6 = np.full((m['n'], 12), np.nan)
    r6[ok] = pct(m['p6r'][ok])
    raptor = 0.65 * pct(m['p384']) + 0.35 * pct(m['p224'])
    compact = np.where(ok[:, None], (r6 + pct(m['s6'])) / 2, pct(m['s6']))
    assert np.abs(out - (0.5 * raptor + 0.5 * compact)).max() < 1e-6


def expect_error(**kw):
    try:
        scenario(**kw)
    except RuntimeError as e:
        return str(e)
    raise AssertionError('no error for %s' % kw)


def test_gates():
    assert 'R6 failed' in expect_error(fail_r6=0.05)
    assert 'S6 disagrees' in expect_error(unrelated_s6=True)


def test_mavit_family():
    out, m = scenario(mavit=True, mavit_fail=(7,))
    ok = np.arange(m['n']) != 7
    rm = np.full((m['n'], 12), np.nan)
    rm[ok] = pct(m['pm'][ok])
    raptor = 0.65 * pct(m['p384']) + 0.35 * pct(m['p224'])
    compact = (pct(m['p6r']) + pct(m['s6'])) / 2
    want = np.where(ok[:, None], 0.425 * raptor + 0.425 * compact + 0.15 * rm, (raptor + compact) / 2)
    assert np.abs(out - want).max() < 1e-6, np.abs(out - want).max()


if __name__ == '__main__':
    tests = [test_weights, test_weights_with_r7, test_member_failure_dropped, test_gates]
    if 'B' in sys.argv:
        tests.append(test_mavit_family)
    for t in tests:
        t()
        print('PASS', t.__name__)
