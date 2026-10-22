# Variant A fusion.  Weights: see the rule at the top of the notebook; nothing here is fitted.
import csv
import numpy as np
import pandas as pd
from v33_fusion import rank_on, weighted_available
from v33_io import LABELS, UID, read_aligned, read_status, write_submission

(WORK / 'submission.csv').unlink(missing_ok=True)
W_FAMILY = 0.5
RAPTOR_W = {'Raptor384': 0.65, 'Raptor224': 0.35}
COMPACT = ['R6', 'R7', 'S6'] if globals().get('PRED_R7') is not None else ['R6', 'S6']
COMPACT_W = {k: 1 / len(COMPACT) for k in COMPACT}

test_ids = pd.read_csv(DATA / 'test.csv', dtype={UID: str})[UID].tolist()
n = len(test_ids)
assert n and n == len(set(test_ids))


def compact_scored(pred_dir):
    # the compact worker fills an unscored study with the column mean: read which studies it really scored
    ok = set()
    for f in Path(pred_dir).glob('worker*/predictions.csv'):
        with open(f, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                try:
                    v = [float(r[L]) for L in LABELS]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(0. <= x <= 1. for x in v):
                    ok.add(r[UID])
    return np.array([u in ok for u in test_ids])


P, V = {}, {}
P['Raptor384'] = read_aligned(WORK / 'raptor_raw.csv', test_ids)
V['Raptor384'] = ~(P['Raptor384'] == 0.5).all(axis=1)   # Raptor leaves a failed study at 0.5 in every column
_, V['Raptor224'], v224m = read_status(WORK / 'raptor224_status.csv', test_ids)
P['Raptor224'] = read_aligned(WORK / 'raptor224_plain.csv', test_ids)
P['Raptor224'][v224m] = read_aligned(WORK / 'raptor224_raw.csv', test_ids)[v224m]   # plain+mirror mean where both ran
P['R6'], V['R6'] = read_aligned(WORK / 'r6_sub.csv', test_ids), compact_scored(PRED_R6)
if 'R7' in COMPACT:
    P['R7'], V['R7'] = read_aligned(WORK / 'r7_sub.csv', test_ids), compact_scored(PRED_R7)
if not S6_OK:
    raise RuntimeError('S6 stage did not complete; no submission written')
s6 = pd.read_csv(S6_PROBS, dtype={UID: str}).set_index(UID).loc[test_ids, LABELS].to_numpy(np.float64)
V['S6'] = np.isfinite(s6).all(axis=1) & ((s6 >= 0) & (s6 <= 1)).all(axis=1)
P['S6'] = np.where(np.isfinite(s6), s6, 0.5)

W_MAVIT = 0.15 if globals().get('MAVIT_OK') else 0.0     # third, independent family (variant B only)
if W_MAVIT:
    P['MaViT'] = read_aligned(WORK / 'mavit_sub.csv', test_ids)
    _ok = pd.read_csv(WORK / 'mavit_status.csv', dtype={UID: str}).set_index(UID).loc[test_ids, 'ok'].to_numpy(bool)
    V['MaViT'] = _ok & ~(P['MaViT'] == 0).all(axis=1)        # the source writes all-zero rows for a failed study

for name, v in V.items():
    if (~v).mean() > MAX_FAILURE_FRACTION:
        raise RuntimeError(f'{name} failed on {(~v).sum()} of {n} studies; inspect its stage log')


def rho(a, b, m):
    if m.sum() < 20:
        return None
    return [float(pd.Series(a[m, j]).corr(pd.Series(b[m, j]), method='spearman')) for j in range(len(LABELS))]


# Same-family agreement gate (the four-model source's S6 gate), against the first compact member.  Only with >= 20 studies.
REF = COMPACT[0]
gates = {}
for name in COMPACT[1:]:
    r = rho(P[name], P[REF], V[name] & V[REF])
    gates[name] = None if r is None else dict(zip(LABELS, [round(x, 4) for x in r]))
    if r is not None and (np.nanmean(r) < 0.6 or np.nanmin(r) < 0.3):
        raise RuntimeError(f'{name} disagrees with {REF} (mean {np.nanmean(r):.3f}, min {np.nanmin(r):.3f}); not blended')
print('FAMILY_GATES vs', REF, json.dumps(gates), flush=True)

raptor = weighted_available([(w, rank_on(P[k], V[k])) for k, w in RAPTOR_W.items()])
compact = weighted_available([(w, rank_on(P[k], V[k])) for k, w in COMPACT_W.items()])
fam = [(W_FAMILY * (1 - W_MAVIT), raptor), ((1 - W_FAMILY) * (1 - W_MAVIT), compact)]
if W_MAVIT:
    fam.append((W_MAVIT, rank_on(P['MaViT'], V['MaViT'])))
final = weighted_available(fam)
final[np.isnan(final)] = 0.5                              # a study every member failed on (cannot pass the 3 % gates in bulk)
assert np.isfinite(final).all() and ((final >= 0) & (final <= 1)).all()

# Diagnostics only (prediction agreement, never an AUC estimate): rank correlation of each member with the blend.
diag = {}
for name in P:
    r = rho(P[name], final, V[name])
    diag[name] = None if r is None else round(float(np.nanmean(r)), 4)
seconds = {'raptor384': RAPTOR_SECONDS, 'raptor224': RAPTOR224_SECONDS, 'r6': R6_SECONDS, 's6': S6_SECONDS or 0}
if 'R7' in COMPACT:
    seconds['r7'] = R7_SECONDS
if W_MAVIT:
    seconds['mavit'] = MAVIT_SECONDS
seconds = {k: round(v, 1) for k, v in seconds.items()}
seconds['total'] = round(time.time() - NB_T0, 1)
receipt = {'variant': 'A', 'members': list(P), 'rehearsal_n': REHEARSAL_N, 'rows': n,
           'weights': {'family': W_FAMILY, 'mavit': W_MAVIT, 'raptor': RAPTOR_W, 'compact': COMPACT_W},
           'failed': {k: int((~v).sum()) for k, v in V.items()}, 'mirror224_failed': int((V['Raptor224'] & ~v224m).sum()),
           'gates_vs_' + REF: gates, 'mean_spearman_member_vs_blend': diag, 'seconds': seconds}
(WORK / 'a5_receipt.json').write_text(json.dumps(receipt, indent=1, allow_nan=False))
out = write_submission(WORK / 'submission.csv', test_ids, final)
print('A_COMPLETE', json.dumps(receipt), flush=True)
display(out.head())
