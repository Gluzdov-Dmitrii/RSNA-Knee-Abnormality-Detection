"""CPU live-DICOM contract tests using synthetic pixels and a mocked classifier."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if 'timm' not in sys.modules and importlib.util.find_spec('timm') is None:
    stub = types.ModuleType('timm')
    def _no_backbone(*args, **kwargs):
        raise AssertionError('Real backbone construction is forbidden in this test')
    stub.create_model = _no_backbone
    sys.modules['timm'] = stub
import infer
import geometry
from common import ARCH, GEOMETRY, TARGETS, UID


class MaskCountModel(nn.Module):
    """Known per-study logits; verifies decoded missing slots reach the model."""
    def forward(self, pixels, mask):
        if pixels.shape[1:] != (18, 3, 224, 224):
            raise AssertionError(f'Unexpected image shape {pixels.shape}')
        if pixels.dtype != torch.uint8:
            raise AssertionError('Inference must supply uint8 pixels')
        if not torch.isfinite(pixels.float()).all():
            raise AssertionError('Nonfinite image')
        return (mask.sum(1).float() / 3 - 2)[:, None].expand(-1, 12)


def write_dicom(path, study_uid, series_uid, index):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = 'MR'
    ds.Rows, ds.Columns = 32, 40
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [5., 5.]
    ds.ImageOrientationPatient = [1., 0., 0., 0., 1., 0.]
    ds.ImagePositionPatient = [0., 0., float(index * 2)]
    # Deliberately misleading fallback order and reversed filenames.
    ds.InstanceNumber = 100 - index
    ds.RescaleSlope = 2.
    ds.RescaleIntercept = -10.
    y, x = np.indices((32, 40))
    ds.PixelData = (x * x + 7 * y + 19 * index).astype('<u2').tobytes()
    ds.save_as(path, enforce_file_format=True)


class LiveInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.uids = [generate_uid() for _ in range(3)]
        rows = []
        for n, uid in enumerate(self.uids, start=1):
            for _, plane, fluid in geometry.SLOTS[:n]:
                series_uid = generate_uid()
                directory = self.root / 'test_series' / uid / series_uid
                directory.mkdir(parents=True)
                for index in range(7):
                    write_dicom(directory / f'{6-index}.dcm', uid, series_uid, index)
                rows.append({UID: uid, 'SeriesInstanceUID': series_uid,
                             'Anatomical_Plane': plane, 'Fluid_Sensitive': fluid})
        self.series = pd.DataFrame(rows)
        self.series.to_csv(self.root / 'test_series.csv', index=False)
        self.checkpoint = self.root / 'weights.pt'
        self.checkpoint.write_bytes(b'synthetic-checkpoint-for-hash-only')
        self.checkpoint_dict = {'geometry': GEOMETRY, 'targets': TARGETS,
                                'arch': ARCH, 'res': 224, 'model': {}}

    def run_order(self, order, sample_order=None):
        pd.DataFrame({UID: order}).to_csv(self.root / 'test.csv', index=False)
        frame = pd.DataFrame({UID: order if sample_order is None else sample_order})
        for target in TARGETS:
            frame[target] = .5
        frame.to_csv(self.root / 'sample_submission.csv', index=False)
        with patch.object(infer, 'KneeModel', return_value=MaskCountModel()), \
             patch.object(infer.torch, 'load', return_value=self.checkpoint_dict):
            return infer.run(self.root, self.checkpoint, self.root / 'result.csv', device='cpu')

    def test_arbitrary_count_sample_order_and_missing_slots(self):
        for order in ([self.uids[2], self.uids[0], self.uids[1]], [self.uids[1]],
                      [self.uids[1], self.uids[2]]):
            with self.subTest(count=len(order)):
                actual = self.run_order(order)
                self.assertEqual(actual[UID].tolist(), order)
                self.assertEqual(actual.columns.tolist(), [UID] + TARGETS)
                expected = torch.sigmoid(torch.tensor([self.uids.index(uid) - 1.
                                                       for uid in order])).numpy()
                np.testing.assert_allclose(actual[TARGETS],
                    np.repeat(expected[:, None], 12, axis=1), rtol=1e-6)
                self.assertTrue(np.isfinite(actual[TARGETS]).all().all())
                saved = pd.read_csv(self.root / 'result.csv', dtype={UID: str})
                self.assertEqual(saved[UID].tolist(), order)
                audit = json.loads((self.root / 'result.audit.json').read_text())
                self.assertEqual(audit['n_studies'], len(order))
                self.assertGreater(audit['decode_audit']['absent_slots'], 0)
                self.assertGreater(audit['decode_audit']['slices_read'], 0)

    def test_missing_study_metadata_fails_without_submission(self):
        with self.assertRaises(ValueError):
            self.run_order([generate_uid()])
        self.assertFalse((self.root / 'result.csv').exists())

    def test_duplicate_test_uids_fail(self):
        # Here both files contain duplicates; authoritative test IDs must be rejected.
        with self.assertRaises(ValueError):
            self.run_order([self.uids[0], self.uids[0]])

    def test_hidden_remount_uses_test_ids_with_stale_sample(self):
        # sample_submission.csv may remain the three visible placeholders while
        # Kaggle replaces test.csv and test_series.csv for scoring.
        stale_sample = [generate_uid() for _ in range(3)]
        order = [self.uids[2], self.uids[0]]
        actual = self.run_order(order, sample_order=stale_sample)
        self.assertEqual(actual[UID].tolist(), order)
        self.assertEqual(len(actual), 2)
        expected = torch.sigmoid(torch.tensor([1., -1.])).numpy()
        np.testing.assert_allclose(actual[TARGETS],
            np.repeat(expected[:, None], 12, axis=1), rtol=1e-6)

    def test_synthetic_geometry_matches_original_cache_pipeline(self):
        original_path = HERE.parents[1] / 'experiments/cache_budget/recheck/pipeline.py'
        spec = importlib.util.spec_from_file_location('original_cache_pipeline_test', original_path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        uid = self.uids[0]
        frame = self.series[self.series[UID] == uid]
        records = geometry.load_series(self.root, frame, Counter(), [GEOMETRY],
                                       series_directory='test_series')
        record = records[geometry.SLOTS[0][0]]
        np.testing.assert_array_equal(geometry.render_slot(record, GEOMETRY, Counter()),
                                      original.render_slot(record, GEOMETRY, Counter()))
        paths = list((self.root / 'test_series' / uid / frame.iloc[0].SeriesInstanceUID).glob('*.dcm'))
        ours, spacing = geometry.order_files(paths, Counter())
        theirs, old_spacing = original.order_files(paths, Counter())
        self.assertEqual(ours, theirs)
        self.assertEqual(spacing, old_spacing)
        self.assertEqual([path.name for path in ours], [f'{6-i}.dcm' for i in range(7)])
        for n in (1, 2, 7, 25):
            self.assertEqual(geometry.sample_indices(n, 9, GEOMETRY['window']),
                             original.sample_indices(n, 9, GEOMETRY['window']))


if __name__ == '__main__':
    unittest.main()
