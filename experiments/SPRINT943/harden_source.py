"""Strict release gates for the audited Speedy Raptors notebook.

Only failure handling and receipts change. Successful predictions, sampling,
AMP, checkpoints, weights and blend arithmetic are preserved. V34 has four
CoAt families; the separately audited latest export has three.
"""
from __future__ import annotations

import ast
import copy
import hashlib


class HardeningError(ValueError):
    """The notebook no longer matches the audited source contracts."""


def _replace(cells, old, new, label):
    matches = [(i, ''.join(c['source']).count(old)) for i, c in enumerate(cells)
               if c['cell_type'] == 'code']
    total = sum(n for _, n in matches)
    if total != 1:
        raise HardeningError(f'{label}: expected one anchor, found {total}')
    i = next(i for i, n in matches if n)
    cells[i]['source'] = ''.join(cells[i]['source']).replace(old, new)


STRICT_HELPERS = '''
# SPRINT943: explicit failures must never become plausible ranked predictions.
_SPRINT_FATAL_EVENTS = {
    'coverage_mismatch', 'slot_no_files', 'slot_all_decode_failed',
    'slot_read_error', 'slot_unfilled', 'a5_slot_no_files',
    'a5_slot_all_crops_failed', 'a5_study_failed',
    'raptor_acquisition_empty', 'raptor_study_failed', 'raptor_neutral_fill',
    'dino_member_nonfinite_neutral', 'dino_member_public_nonfinite_neutral',
    'dino_jitter_fp16_nonfinite_dropped', 'a5_nonfinite_neutral',
    'rad_invalid_values_repaired', 'coat_family_unavailable',
    'coat_family_partial', 'coat_member_probability_unavailable',
    'raptor_preparation_identities_incomplete',
}

def sprint_validate_events(events):
    failures = [e for e in events if e.get('kind') in _SPRINT_FATAL_EVENTS
                or (e.get('kind') == 'a5_decode_failure' and int(e.get('rendered', 0)) <= 0)
                or (str(e.get('kind', '')).startswith('coat_') and
                    any(t in str(e.get('kind', '')) for t in
                        ('child_failed', 'output_rejected', 'fallback_studies')))]
    if failures:
        raise RuntimeError('SPRINT943 rejects degraded inference: ' + repr(failures[:8]))

def sprint_validate_coat_receipt(receipt, expected_members, study_count):
    members = receipt.get('members', [])
    if len(members) != len(set(members)) or set(members) != set(expected_members):
        raise RuntimeError('Incomplete CoAt family: ' + repr(members))
    if receipt.get('family_reduction') != 'rank_of_member_probability_mean':
        raise RuntimeError('CoAt probability-mean reduction was not used')
    if receipt.get('study_count') != study_count:
        raise RuntimeError('CoAt receipt study count mismatch')
    weights = receipt.get('within_coat', {})
    if set(weights) != set(expected_members) or any(
            abs(float(weights[m]) - 1.0 / len(expected_members)) > 1e-12
            for m in expected_members):
        raise RuntimeError('CoAt receipt weights mismatch')
    if receipt.get('inner_rerank') is not False or receipt.get('private_alpha') != .4:
        raise RuntimeError('CoAt inner contract mismatch')
    return receipt

def sprint_validate_probability_payload(uids, prob, expected_ids, labels, payload_labels=None):
    import numpy as np
    ids = [str(u) for u in uids]
    expected = [str(u) for u in expected_ids]
    if len(ids) != len(set(ids)) or len(ids) != len(expected) or set(ids) != set(expected):
        raise RuntimeError('CoAt probability payload UID mismatch')
    if payload_labels is not None and list(map(str, payload_labels)) != list(labels):
        raise RuntimeError('CoAt probability payload label order mismatch')
    x = np.asarray(prob)
    if x.shape != (len(ids), len(labels)) or not np.isfinite(x).all():
        raise RuntimeError('CoAt probability payload shape/nonfinite mismatch')
    if ((x < 0) | (x > 1)).any():
        raise RuntimeError('CoAt probability payload outside [0,1]')

def sprint_release_gate(work, expected_ids, expected_members, events):
    import json
    from pathlib import Path
    import numpy as np
    work = Path(work)
    sprint_validate_events(events)
    receipt = json.loads((work / '_coat_raptor_blend_receipt.json').read_text())
    sprint_validate_coat_receipt(receipt, expected_members, len(expected_ids))
    specs = {
        'resgated_top3': ('_coat_arm_receipt.json', 3, [4, 6, 8]),
        'global96_top3': ('_coat_g96_receipt.json', 3, [16, 23, 18]),
        'd4_swa3': ('d4_input/d4.receipt.json', 1, None),
        'repairv1_top3': ('_coat_repairv1_receipt.json', 3, [12, 7, 11]),
    }
    children = {}
    for member in expected_members:
        filename, count, epochs = specs[member]
        child = json.loads((work / filename).read_text())
        # The V34 repair runtime uses .get('fallback_studies', 0), whereas
        # the other three source wrappers require the field explicitly.
        fallback = child.get('fallback_studies', 0) if member == 'repairv1_top3' else child.get('fallback_studies')
        if child.get('models') != count or fallback != 0:
            raise RuntimeError(member + ': incomplete models or fallback studies')
        if epochs is not None and [c.get('epoch') for c in child.get('checkpoints', [])] != epochs:
            raise RuntimeError(member + ': checkpoint epoch mismatch')
        for shard in child.get('shards', []):
            if int(shard.get('fallback_studies', 0)) or int(shard.get('preparation_warnings', 0)):
                raise RuntimeError(member + ': shard preparation degradation')
        children[member] = {'models': count, 'fallback_studies': 0,
                            'receipt': filename, 'epochs': epochs}
    meta = json.loads((work / 'diagnostics/a5_metadata_applicability.json').read_text())
    decoded = json.loads((work / 'diagnostics/a5_applicability.json').read_text())
    if set(meta['no_matching_acquisition_uids']) != set(decoded['unavailable_uids']):
        raise RuntimeError('A5 decoded absence differs from acquisition metadata')
    with np.load(work / 'raptor_raw.npz', allow_pickle=False) as z:
        if z['study_uids'].astype(str).tolist() != list(expected_ids):
            raise RuntimeError('Raptor raw UID order mismatch')
        raw = z['raw_probabilities']
        if raw.shape != (4, len(expected_ids), 12) or not np.isfinite(raw).all():
            raise RuntimeError('Incomplete four-view Raptor raw predictions')
        if ((raw < 0) | (raw > 1)).any():
            raise RuntimeError('Raptor raw predictions outside [0,1]')
    for path in (work / 'd4_input/study_input_receipts').glob('input_*.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip() and any('fallback' in item for item in json.loads(line).get('slots', [])):
                raise RuntimeError('D4 dense preparation fell back to another input grid')
    return {'status': 'VERIFIED', 'coats': children, 'family': receipt,
            'acquisition_absence_checked': True, 'raw_raptor_checked': True}
'''


def harden_notebook(nb: dict) -> dict:
    """Return an independent hardened notebook; reject changed source anchors."""
    out = copy.deepcopy(nb)
    cells = out['cells']
    original = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    if 'SPRINT943: explicit failures' in original:
        raise HardeningError('Notebook already hardened; use the immutable source')
    repair = "_members.append(('repairv1_top3'" in original
    expected = ['resgated_top3', 'global96_top3', 'd4_swa3']
    if repair:
        expected.append('repairv1_top3')
    _replace(cells, 'def rsna_asset_preflight():', STRICT_HELPERS + '\n\ndef rsna_asset_preflight():', 'strict helpers')

    # Unrecoverable model output errors stop immediately, after existing retries.
    _replace(cells, '            pred = np.where(np.isfinite(pred), pred, 0.5)',
             "            raise WeightsError('DINO raw prediction remains nonfinite after retry')", 'DINO raw')
    _replace(cells, '            public_pred = np.where(np.isfinite(public_pred), public_pred, 0.5)',
             "            raise WeightsError('DINO retained prediction remains nonfinite after retry')", 'DINO public')
    _replace(cells, '        got = np.where(_bad, 0.5, got)',
             "        raise RuntimeError('A5 prediction remains nonfinite after full-precision retry')", 'A5 raw')
    _replace(cells, '        frame[_RAD_LABELS] = _rad_np.clip(_rad_np.where(_rad_np.isfinite(values), values, 0.5), 0, 1)',
             "        raise RuntimeError('Rad output is nonfinite or outside [0,1]')", 'Rad raw')
    _replace(cells, '            _arr[_bad] = _ke_np.where(_ke_np.isfinite(_fill), _fill, .5)',
             "            raise RuntimeError('Raptor arm has missing/nonfinite raw predictions')", 'Raptor raw')
    _replace(cells, '                imgs[i] = 0; msks[i] = 0',
             "                raise RuntimeError('A5 acquired-study preparation failed') from e", 'A5 preparation')

    # Child execution and result acceptance cannot drop mandatory families.
    kinds = ['resgated', 'g96', 'd4'] + (['rv1'] if repair else [])
    for name in kinds:
        _replace(cells, f'        _coat_{name}_ok = False',
                 f'        raise RuntimeError("Required CoAt branch {name} failed") from _{name if name != "resgated" else "coat"}_exc',
                 f'{name} child failure')
    for name in ['resgated', 'global96', 'd4'] + (['repairv1'] if repair else []):
        anchor = f"            rsna_event('coat_{name}_output_rejected', error=f'{{type(_exc).__name__}}: {{str(_exc)[:500]}}')"
        _replace(cells, anchor, anchor + '\n            raise', f'{name} output rejection')
    _replace(cells, '    if not _members:\n',
             f"    if [m for m, _ in _members] != {expected!r}:\n        raise RuntimeError('Incomplete mandatory CoAt family')\n    if not _members:\n", 'CoAt completeness')
    anchor = "            rsna_event('coat_member_probability_unavailable', member=_name,\n                       error=f'{type(_exc).__name__}: {str(_exc)[:300]}')"
    _replace(cells, anchor, anchor + '\n            raise', 'CoAt probability required')
    anchor = "                prob = _np.asarray(z['raw_probabilities'], dtype=_np.float64).mean(axis=0)"
    _replace(cells, anchor, anchor + "\n            sprint_validate_probability_payload(uids, prob, pub.StudyInstanceUID.astype(str).tolist(), lab, z['labels'].astype(str).tolist() if 'labels' in z.files else None)", 'CoAt payload identity')

    # Dense-grid fallback would silently alter the evaluated model inputs.
    anchor = "                             '        return _original_train_faithful_eval_specs(study)\\n'"
    _replace(cells, anchor, "                             '        raise RuntimeError(\"Required dense CoAt preparation failed\") from exc\\n'", 'dense CoAt grid')
    anchor = "                          '    input_audit = [{\"fallback\": type(_dense_exc).__name__ + \": \" + str(_dense_exc)}]\\n'"
    _replace(cells, anchor, "                          '    raise RuntimeError(\"Required dense D4 preparation failed\") from _dense_exc\\n'", 'dense D4 grid')

    # Source metadata proves genuine A5 absence; decode errors must not add rows.
    anchor = "rsna_phase('a5', 'COMPLETE', applicable=int(_a5_eligible.sum()), absent=int((~_a5_eligible).sum()))"
    _replace(cells, anchor, "sprint_validate_events(_RSNA_AUDIT['events'])\n" + anchor, 'A5 phase gate')
    anchor = "    rsna_event('raptor_preparation_identities_incomplete', missing=len(_expected_preparations - _ke_input_ids), extra=len(_ke_input_ids - _expected_preparations))"
    _replace(cells, anchor, anchor + "\n    raise RuntimeError('Incomplete Raptor preparation identity set')", 'Raptor preparation coverage')
    anchor = "rsna_save_predictions('final',_RSNA_TEST_IDS,_release_df[_RSNA_LABELS].to_numpy(),_RSNA_LABELS)"
    _replace(cells, anchor, f"_SPRINT_RELEASE = sprint_release_gate('/kaggle/working', _RSNA_TEST_IDS, {expected!r}, _RSNA_AUDIT['events'])\n" + anchor, 'final actual gate')
    anchor = "for cache_file in list(_RSNA_CACHE_FILES):Path(cache_file).unlink(missing_ok=True)"
    _replace(cells, anchor,
             "_RSNA_AUDIT['coats'] = _SPRINT_RELEASE['coats']\n"
             "_RSNA_AUDIT['family_mix'] = _SPRINT_RELEASE['family']['within_coat']\n"
             "_RSNA_AUDIT['strict_release'] = _SPRINT_RELEASE\n" + anchor, 'truthful final receipt')

    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source, filename=f'cell_{index}')
            compile(source, f'cell_{index}', 'exec')
            cell['source'] = source.splitlines(keepends=True)
    out.setdefault('metadata', {})['sprint943_hardening'] = {
        'version': 1, 'expected_coat_members': expected,
        'successful_prediction_math_changed': False,
        'source_code_sha256': hashlib.sha256(original.encode()).hexdigest(),
    }
    return out
