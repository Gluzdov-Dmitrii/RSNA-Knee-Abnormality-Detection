"""CPU regression tests for failure gates; no images, weights or GPUs needed."""
import copy
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from harden_source import HardeningError, STRICT_HELPERS, harden_notebook


HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source_v34/bend-the-knee-to-speedy-raptors-the-original.ipynb'
MEMBERS = ['resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3']
NS = {}
exec(compile(STRICT_HELPERS, 'strict_helpers', 'exec'), NS)


def code(nb):
    return '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')


def valid_receipt():
    return {'members': MEMBERS.copy(), 'family_reduction': 'rank_of_member_probability_mean',
            'study_count': 2, 'within_coat': dict.fromkeys(MEMBERS, .25),
            'inner_rerank': False, 'private_alpha': .4}


class HardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(SOURCE.read_text(encoding='utf-8'))
        cls.hardened = harden_notebook(cls.source)

    def test_nonmutating_and_compiles(self):
        original = copy.deepcopy(self.source)
        result = harden_notebook(self.source)
        self.assertEqual(original, self.source)
        self.assertEqual(MEMBERS, result['metadata']['sprint943_hardening']['expected_coat_members'])
        for i, c in enumerate(result['cells']):
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), str(i), 'exec')

    def test_drift_and_duplicate_application_rejected(self):
        broken = copy.deepcopy(self.source)
        for c in broken['cells']:
            if c['cell_type'] == 'code':
                c['source'] = ''.join(c['source']).replace('def rsna_asset_preflight():', 'def changed_preflight():')
        with self.assertRaises(HardeningError):
            harden_notebook(broken)
        with self.assertRaises(HardeningError):
            harden_notebook(self.hardened)

    def test_success_math_preserved(self):
        before, after = code(self.source), code(self.hardened)
        # Each successful prediction expression is unchanged, including the
        # source version's unusual per-target and probability/rank decisions.
        anchors = [
            "_coatnet_weight = {label: 0.60 for label in _blend_labels}",
            "    'Lateral Meniscus': 1.00,",
            "A5_W = 0.52", "AMP_PREF = 'bf16'",
            '        private_rank = _mean.rank(method=\'average\', pct=True)',
            '    hybrid[lab] = (1.0 - private_alpha) * public_rank + private_alpha * private_rank',
            "    frontier_ids, frontier_acc = _combine(public_frontier_members)",
            '    quotas = _dense_allocate(capacities, [s[2] for s in slots], 96)',
        ]
        for anchor in anchors:
            self.assertIn(anchor, before)
            self.assertEqual(before.count(anchor), after.count(anchor), anchor)

    def test_generated_dense_wrappers_compile(self):
        trees = [ast.parse(''.join(c['source'])) for c in self.hardened['cells']
                 if c['cell_type'] == 'code']
        functions = {n.name: n for t in trees for n in t.body if isinstance(n, ast.FunctionDef)}
        patch_assignment = next(n for n in ast.walk(functions['_asset_write_coat_runtime'])
                                if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id == 'patch' for v in n.targets))
        expression = ast.Expression(patch_assignment.value)
        generated = eval(compile(expression, 'generated_coat', 'eval'), {'helper_source': ''})
        compile(generated, 'generated_coat', 'exec')
        d4_parse = next(n for n in ast.walk(functions['_asset_patch_d4'])
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == 'parse' and n.args and isinstance(n.args[0], ast.Constant)
                        and str(n.args[0].value).startswith('try:'))
        generated = ast.literal_eval(d4_parse.args[0])
        compile(generated, 'generated_d4', 'exec')
        self.assertIn('raise RuntimeError', generated)

    def test_allowed_absence_and_retry_events(self):
        allowed = [
            {'kind': 'empty_study_rows'}, {'kind': 'a5_empty_study'},
            {'kind': 'a5_no_metadata'}, {'kind': 'scratch_fallback'},
            {'kind': 'a5_fp16_nonfinite_retry_fp32'},
            {'kind': 'partial_slice_decode'}, {'kind': 'slot_shape_mismatch'},
            {'kind': 'a5_order_unreadable_dropped'}, {'kind': 'raptor_order_unreadable_skipped'},
            {'kind': 'a5_decode_failure', 'rendered': 15, 'wanted': 16},
        ]
        NS['sprint_validate_events'](allowed)

    def test_degraded_events_rejected(self):
        for kind in ['slot_read_error', 'a5_decode_failure',
                     'raptor_neutral_fill', 'coat_repairv1_child_failed',
                     'coat_global96_output_rejected', 'coat_member_probability_unavailable']:
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                NS['sprint_validate_events']([{'kind': kind}])

    def test_family_contract(self):
        r = valid_receipt()
        self.assertIs(r, NS['sprint_validate_coat_receipt'](r, MEMBERS, 2))
        changes = [('members', MEMBERS[:-1]), ('family_reduction', 'rank_sum_fallback_probabilities_unavailable'),
                   ('study_count', 1), ('within_coat', dict.fromkeys(MEMBERS, .5))]
        for key, value in changes:
            bad = copy.deepcopy(r)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                NS['sprint_validate_coat_receipt'](bad, MEMBERS, 2)

    def test_raw_payload(self):
        validate = NS['sprint_validate_probability_payload']
        labels = ['a', 'b']
        x = np.array([[.2, .3], [.4, .6]])
        validate(['u2', 'u1'], x, ['u1', 'u2'], labels, labels)
        for ids, values, cols in [(['u1', 'u1'], x, labels),
                                  (['u1', 'u2'], x * np.nan, labels),
                                  (['u1', 'u2'], x * 3, labels),
                                  (['u1', 'u2'], x, labels[::-1])]:
            with self.assertRaises(RuntimeError):
                validate(ids, values, ['u1', 'u2'], labels, cols)

    def test_actual_nested_npz_reader_executes_inside_context(self):
        trees = [ast.parse(''.join(c['source'])) for c in self.hardened['cells']
                 if c['cell_type'] == 'code']
        reader = next(n for t in trees for n in ast.walk(t)
                      if isinstance(n, ast.FunctionDef) and n.name == '_coat_member_probability')
        # Execute the actual transformed notebook helper, not a synthetic copy.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'member.npz'
            ids, labels = ['u2', 'u1'], ['ACL', 'MCL']
            values = np.array([[.2, .3], [.7, .8]])
            np.savez(path, study_uids=np.asarray(ids), labels=np.asarray(labels),
                     probability_mean=values)
            namespace = {'_np': np, '_pd': pd, '_PROB_FILES': {'fixture': path},
                         'pub': pd.DataFrame({'StudyInstanceUID': ['u1', 'u2']}),
                         'lab': labels, 'sprint_validate_probability_payload': NS['sprint_validate_probability_payload']}
            module = ast.Module(body=[reader], type_ignores=[])
            exec(compile(module, 'actual_notebook_helper', 'exec'), namespace)
            result = namespace['_coat_member_probability']('fixture')
            np.testing.assert_array_equal(result.to_numpy(), values[::-1])
            # Exercise the source's alternative raw-probability mean path too.
            np.savez(path, study_uids=np.asarray(ids), labels=np.asarray(labels),
                     raw_probabilities=np.stack([values, values]))
            result = namespace['_coat_member_probability']('fixture')
            np.testing.assert_array_equal(result.to_numpy(), values[::-1])
            np.savez(path, study_uids=np.asarray(['u1', 'u1']), labels=np.asarray(labels),
                     probability_mean=values)
            with self.assertRaisesRegex(RuntimeError, 'UID mismatch'):
                namespace['_coat_member_probability']('fixture')

    def test_release_fixture_and_metadata_decode_disagreement(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / 'diagnostics').mkdir()
            (work / 'd4_input/study_input_receipts').mkdir(parents=True)
            def write(path, value):
                (work / path).write_text(json.dumps(value))
            write('_coat_raptor_blend_receipt.json', valid_receipt())
            for file, epochs in [('_coat_arm_receipt.json', [4, 6, 8]),
                                 ('_coat_g96_receipt.json', [16, 23, 18]),
                                 ('_coat_repairv1_receipt.json', [12, 7, 11])]:
                write(file, {'models': 3, 'fallback_studies': 0,
                             'checkpoints': [{'epoch': e} for e in epochs]})
            write('d4_input/d4.receipt.json', {'models': 1, 'fallback_studies': 0})
            write('diagnostics/a5_metadata_applicability.json', {'no_matching_acquisition_uids': ['u2']})
            write('diagnostics/a5_applicability.json', {'unavailable_uids': ['u2']})
            np.savez(work / 'raptor_raw.npz', study_uids=np.array(['u1', 'u2']),
                     raw_probabilities=np.full((4, 2, 12), .5))
            fn = NS['sprint_release_gate']
            self.assertEqual('VERIFIED', fn(work, ['u1', 'u2'], MEMBERS, [])['status'])
            write('diagnostics/a5_applicability.json', {'unavailable_uids': ['u1', 'u2']})
            with self.assertRaisesRegex(RuntimeError, 'acquisition metadata'):
                fn(work, ['u1', 'u2'], MEMBERS, [])
            write('diagnostics/a5_applicability.json', {'unavailable_uids': ['u2']})
            write('_coat_arm_receipt.json', {'models': 3, 'fallback_studies': 1})
            with self.assertRaisesRegex(RuntimeError, 'fallback studies'):
                fn(work, ['u1', 'u2'], MEMBERS, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
