"""Regression checks for target/context separation and state initialization."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from prepare import prepare, SPLITS
from dynamix.utilities.dataset import Dataset
from dynamix.training.teacher_forcing import predict_sequence_using_gtf


class ContextExperimentTests(unittest.TestCase):
    def make_source(self, directory, repeats=2):
        directory.mkdir()
        values = np.repeat(np.arange(-1, 1.01, 0.25), repeats).astype(np.float32)
        x = np.empty((len(values), 100, 3), dtype=np.float32)
        x[:, :, 0] = np.arange(len(values))[:, None]
        x[:, :, 1] = np.arange(100)[None, :]
        x[:, :, 2] = values[:, None]
        np.save(directory / 'trajectories.npy', x)
        np.save(directory / 'trajectory_phi.npy', np.broadcast_to(values[:, None, None], (len(values), 100, 1)))
        np.save(directory / 'trajectory_phi_raw.npy', np.broadcast_to((1.75 + values * .35)[:, None, None], (len(values), 100, 1)))
        return x

    def verify_preparation(self, repeats):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = self.make_source(root / 'source', repeats)
            output = root / 'prepared'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root / 'source', output, context_length=8, prediction_length=7,
                        test_length=24, windows_per_trajectory=4)
            for split, requested in SPLITS.items():
                fixed, matched = output / split / 'fixed', output / split / 'matched'
                np.testing.assert_array_equal(np.load(fixed / 'data.npy'), np.load(matched / 'data.npy'))
                np.testing.assert_array_equal(np.load(fixed / 'phi.npy'), np.load(matched / 'phi.npy'))
                self.assertEqual(np.load(fixed / 'data.npy').shape[0], 8)  # x0 + 7 transitions
                np.testing.assert_allclose(np.load(fixed / 'context_phi.npy'), -.75)
                self.assertEqual(set(np.unique(np.load(fixed / 'phi.npy'))), set(requested))
                for directory in (fixed, matched):
                    target_series = np.load(directory / 'window_series_indices.npy')
                    data_starts = np.load(directory / 'data_start_indices.npy')
                    ctx_series = np.load(directory / 'context_series_indices.npy')
                    ctx_starts = np.load(directory / 'context_start_indices.npy')
                    data = np.load(directory / 'data.npy')
                    contexts = np.load(directory / 'context.npy')
                    for i, (s, t, cs, ct) in enumerate(zip(target_series, data_starts, ctx_series, ctx_starts)):
                        np.testing.assert_array_equal(data[:, i], x[s, t:t + 8])
                        np.testing.assert_array_equal(contexts[:, i], x[cs, ct:ct + 8])
                        self.assertLessEqual(t + 8, 76)
                        self.assertLessEqual(ct + 8, 76)
                        self.assertTrue(s != cs or ct + 8 <= t or ct >= t + 8)
            test = np.load(output / 'evaluation/test.npy')
            np.testing.assert_array_equal(test, x[:, -24:].transpose(1, 0, 2))
            fixed_test = np.load(output / 'evaluation/fixed_context.npy')
            np.testing.assert_allclose(fixed_test[:, :, 2], -.75)
            self.assertEqual(len(np.unique(test[:, :, 2])), 9)
            with self.assertRaises(FileExistsError):
                prepare(root / 'source', output)

    def test_paired_windows_and_heldout_tail(self):
        self.verify_preparation(repeats=2)

    def test_single_reference_trajectory_cannot_overlap_target(self):
        self.verify_preparation(repeats=1)

    def test_dataset_preserves_clean_target_initial_state(self):
        data = np.arange(36, dtype=np.float32).reshape(4, 3, 3)
        context = np.full((8, 3, 3), -9, dtype=np.float32)
        ds = Dataset(data, context, np.zeros((24, 5, 3)), batch_size=3, noise_level=1,
                     test_context=np.zeros((8, 5, 3)), clean_initial_state=True)
        batch_x, batch_y, _ = ds.sample_batch()
        for i in range(3):
            idx = int(batch_y[0, i, 0].item() - 9) // 3
            np.testing.assert_array_equal(batch_x[0, i].numpy(), data[0, idx])
        with self.assertRaises(ValueError):
            Dataset(data, context, np.zeros((24, 5, 3)), test_context=np.zeros((8, 3, 3)))

    def test_teacher_forcing_uses_target_x0_and_target_phi(self):
        class Recorder(torch.nn.Module):
            phi_dim, M = 1, 3
            B = torch.eye(3)

            def forward(self, z, context, phi_t):
                if not hasattr(self, 'initial'):
                    self.initial = z.clone()
                self.phi = phi_t.clone()
                self.context = context.clone()
                return z + phi_t

        model = Recorder()
        x = torch.ones(4, 2, 3) * 7
        context = torch.ones(8, 2, 3) * -20
        phi = torch.ones(4, 2, 1) * .25
        result = predict_sequence_using_gtf(model, x, context, 1, 10, phi=phi)
        torch.testing.assert_close(model.initial, x[0].t())
        torch.testing.assert_close(model.context, context)
        torch.testing.assert_close(model.phi, phi[0].t())
        torch.testing.assert_close(result[0], x[0] + .25)

    def test_free_rollout_uses_target_initial_point_without_context_scaling(self):
        from evaluate import forecast
        from dynamix.model.dynamix import DynaMix
        from dynamix.model.forecaster import DynaMixForecaster

        model = DynaMix(M=3, N=3, Experts=2, P=2, phi_dim=1)
        with torch.no_grad():
            for expert in model.experts:
                expert.A.fill_(1)
                expert.W.zero_()
                expert.h.zero_()
                expert.C.fill_(1)
        run = {'model': model, 'forecaster': DynaMixForecaster(model)}
        initial = np.array([[3, 4, 5]], dtype=np.float32)
        prediction, weights = forecast(run, np.full((8, 1, 3), -20, dtype=np.float32),
                                       initial, [.25], 4, torch.device('cpu'), 8)
        expected = initial[None] + np.arange(1, 5, dtype=np.float32)[:, None, None] * .25
        np.testing.assert_allclose(prediction, expected, atol=1e-5)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
