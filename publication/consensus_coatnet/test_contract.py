"""CPU-only checks for publication data and model contracts.

Run: python -m unittest discover -s publication/consensus_coatnet -p test_contract.py -v
"""
import sys
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TARGETS, UID, median_scores, read_scores, windows
# Model construction is patched below; timm is unnecessary for these contract tests.
# This does not test installation or real CoAtNet checkpoint compatibility.
if importlib.util.find_spec('timm') is None:
    timm_stub = types.ModuleType('timm')
    def _unexpected_create_model(*args, **kwargs):
        raise AssertionError('A contract test attempted to build a real backbone')
    timm_stub.create_model = _unexpected_create_model
    sys.modules['timm'] = timm_stub
from model import KneeModel, masked_bce


class TinyBackbone(nn.Module):
    """A deterministic image encoder, avoiding pretrained downloads and CoAtNet RAM."""
    num_features = 4

    def forward(self, images):
        channels = images.mean(dim=(-2, -1))
        return torch.cat([channels, channels.square().mean(dim=1, keepdim=True)], dim=1)


def tiny_model():
    torch.manual_seed(2026)
    with patch('model.timm.create_model', return_value=TinyBackbone()):
        return KneeModel(pretrained=False).eval()


class LabelContractTests(unittest.TestCase):
    def test_median_keeps_supplied_scores_and_skips_only_nan(self):
        arrays = [np.array([[np.nan, .28, .5, np.nan]], dtype=np.float32),
                  np.array([[np.nan, .5, .8, .2]], dtype=np.float32),
                  np.array([[np.nan, .9, 0., .8]], dtype=np.float32)]
        actual = median_scores(arrays)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isnan(actual[0, 0]))
        np.testing.assert_allclose(actual[0, 1:], [.5, .5, .5])

    def write_scores(self, directory, uids, score=.2):
        data = {UID: uids}
        data.update({target: [score] * len(uids) for target in TARGETS})
        path = Path(directory) / 'scores.csv'
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def test_duplicate_source_uids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_scores(directory, ['a', 'a'])
            with self.assertRaises(ValueError):
                read_scores(path, ['a'])

    def test_disjoint_uid_table_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_scores(directory, ['unrelated-a', 'unrelated-b'])
            with self.assertRaises(ValueError):
                read_scores(path, ['a', 'b'])

    def test_invalid_scores_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            for bad in (-.01, 1.01, np.inf, -np.inf, 'not-a-number'):
                with self.subTest(score=bad):
                    path = self.write_scores(directory, ['a'], bad)
                    with self.assertRaises((ValueError, TypeError)):
                        read_scores(path, ['a'])

    def test_alignment_is_by_uid_not_row_position(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_scores(directory, ['b', 'a'])
            frame = pd.read_csv(path)
            frame.loc[0, TARGETS] = .8
            frame.to_csv(path, index=False)
            actual = read_scores(path, ['a', 'b', 'missing'])
            np.testing.assert_allclose(actual[0], .2)
            np.testing.assert_allclose(actual[1], .8)
            self.assertTrue(np.isnan(actual[2]).all())


class WindowContractTests(unittest.TestCase):
    def test_triplets_never_cross_slot_or_three_slice_group(self):
        ids = np.arange(54, dtype=np.uint8).reshape(6, 9, 1, 1)
        pixels = np.broadcast_to(ids, (6, 9, 224, 224))
        mask = np.array([True, False, True, False, True, True])
        triplets, present = windows(pixels, mask)
        self.assertEqual(triplets.shape, (18, 3, 224, 224))
        for slot in range(6):
            for group in range(3):
                index = slot * 3 + group
                expected = slot * 9 + group * 3 + np.arange(3)
                np.testing.assert_array_equal(triplets[index, :, 0, 0], expected)
                self.assertEqual(bool(present[index]), bool(mask[slot]))
        self.assertFalse(np.shares_memory(triplets, pixels))

    def test_wrong_geometry_fails(self):
        with self.assertRaises(ValueError):
            windows(np.zeros((6, 3, 224, 224), dtype=np.uint8), np.ones(6, bool))


class ModelContractTests(unittest.TestCase):
    def test_missing_window_features_do_not_change_attention_output(self):
        model = tiny_model()
        features = torch.randn(2, 18, 4)
        mask = torch.ones(2, 18, dtype=torch.bool)
        mask[:, [1, 5, 17]] = False
        changed = features.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 1000
        with torch.no_grad():
            torch.testing.assert_close(model.pool(features, mask), model.pool(changed, mask))
            # Dropping absent windows entirely must have the same result.
            torch.testing.assert_close(model.pool(features, mask),
                model.pool(features[:, mask[0]], torch.ones(2, 15, dtype=torch.bool)))

    def test_missing_pixels_do_not_reach_encoder(self):
        model = tiny_model()
        pixels = torch.randint(0, 256, (2, 18, 3, 8, 8), dtype=torch.uint8)
        mask = torch.ones(2, 18, dtype=torch.bool)
        mask[0, :3] = False
        mask[1, 12:] = False
        changed = pixels.clone()
        changed[~mask] = 255 - changed[~mask]
        with torch.no_grad():
            torch.testing.assert_close(model(pixels, mask), model(changed, mask))

    def test_empty_study_fails(self):
        model = tiny_model()
        mask = torch.ones(2, 18, dtype=torch.bool)
        mask[1] = False
        with self.assertRaises(ValueError):
            model.pool(torch.zeros(2, 18, 4), mask)

    def test_invalid_label_positions_have_zero_gradient(self):
        logits = torch.tensor([[.2, -.3, .5, -.7]], requires_grad=True)
        targets = torch.tensor([[1., float('nan'), .28, float('inf')]])
        loss = masked_bce(logits, targets)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[0, [0, 2]], targets[0, [0, 2]])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[0, [1, 3]], torch.zeros(2))
        self.assertTrue((logits.grad[0, [0, 2]].abs() > 0).all())

    def test_batch_without_labels_fails(self):
        with self.assertRaises(ValueError):
            masked_bce(torch.zeros(1, 12), torch.full((1, 12), float('nan')))


if __name__ == '__main__':
    unittest.main()
