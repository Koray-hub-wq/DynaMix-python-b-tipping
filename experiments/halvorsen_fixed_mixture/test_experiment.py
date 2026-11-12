"""Scientific invariants: mixture, gradient flow, data isolation and exact resume."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch

from common import FixedMixture, SPLITS, Trajectories, load_run
import train
import run
from evaluate import (merge_expert_caches, measure, plot_trajectories,
                      plotting_prediction, rollout_status, trajectory_limits)

torch.set_num_threads(1)


def make_source(directory):
    directory.mkdir()
    values = np.repeat(np.arange(-1, 1.01, .25), 2).astype(np.float32)
    x = np.zeros((len(values), 90, 3), dtype=np.float32)
    x[:, :, 0] = np.arange(len(values))[:, None] / len(values)
    x[:, :, 1] = np.arange(90)[None] / 90
    x[:, :, 2] = values[:, None]
    np.save(directory / 'trajectories.npy', x)
    for name, data in (('trajectory_phi', values), ('trajectory_phi_raw', values * .35 + 1.75)):
        np.save(directory / f'{name}.npy', np.broadcast_to(data[:, None, None], (*x.shape[:2], 1)))
        np.save(directory / f'{name}_constants.npy', data)
    return x


class FixedMixtureTests(unittest.TestCase):
    def test_extreme_float32_axis_limits_do_not_overflow(self):
        extreme = np.finfo(np.float32).max
        values = np.array([[-extreme] * 3, [extreme] * 3], dtype=np.float32)
        with np.errstate(over='raise', invalid='raise'):
            limits = trajectory_limits(np.zeros((2, 3), dtype=np.float32), values)
        self.assertTrue(np.isfinite(list(limits.values())).all())
        self.assertLess(limits['zlim'][0], -float(extreme))

    def test_divergence_diagnostics_preserve_valid_short_horizon_metrics(self):
        t = np.linspace(0, 12, 260)
        truth = np.stack((np.sin(t), np.cos(t), np.sin(2 * t)), axis=1).astype(np.float32)
        prediction = truth.copy()
        prediction[220:, 2] = -np.finfo(np.float32).max
        with np.errstate(over='raise', invalid='raise'):
            row = measure(prediction, truth)
            prefix, note = plotting_prediction(prediction)
        self.assertTrue(row['finite'])
        self.assertTrue(row['diverged'])
        self.assertEqual(row['first_divergence_step'], 221)
        self.assertEqual(row['mse_200'], 0)
        self.assertEqual(row['mase'], 0)
        self.assertIsNone(row['dstsp'])
        self.assertIsNone(row['dh'])
        self.assertEqual(len(prefix), 220)
        self.assertIn('DIVERGED', note)
        prediction[0] = np.nan
        self.assertFalse(rollout_status(prediction)['finite'])
        self.assertEqual(len(plotting_prediction(prediction)[0]), 0)
        prediction[:] = 0
        self.assertFalse(rollout_status(prediction)['diverged'])

    def test_all_nine_plots_survive_huge_nan_and_infinite_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = np.linspace(0, 8, 60)
            target = np.stack((np.sin(t), np.cos(t), np.sin(2 * t)), axis=1).astype(np.float32)
            truth = np.repeat(target[:, None], 9, axis=1)
            prediction = truth.copy()
            prediction[20:, 0] = np.finfo(np.float32).max
            prediction[20:, 1] = -np.finfo(np.float32).max
            prediction[20:, 2] = np.inf
            prediction[:, 3] = np.nan
            prediction[:, 4] = np.inf
            prediction[:, 5] = -np.inf
            prediction[30:, 6] = 1e37
            original = prediction.copy()
            data = SimpleNamespace(values=np.arange(-1, 1.01, .25),
                                   raw_values=1.75 + .35 * np.arange(-1, 1.01, .25))
            with np.errstate(over='raise', invalid='raise'):
                plot_trajectories(Path(tmp), prediction, truth, data,
                                  {'split': 'negative3', 'label': 'divergence_regression'})
            self.assertEqual(len(list(Path(tmp).glob('*.png'))), 11)
            np.testing.assert_array_equal(prediction, original)

    def test_explicit_expert_sum_outputs_and_gradients(self):
        for count in (10, 40):
            torch.manual_seed(8)
            model = FixedMixture(M=6, Experts=count, dtype=torch.float64)
            self.assertFalse(hasattr(model, 'gating_network'))
            self.assertNotIn('weights', dict(model.named_parameters()))
            self.assertTrue(torch.all(model.weights > 0))
            self.assertAlmostEqual(float(model.weights.sum()), 1.0)
            z, phi = torch.randn(6, 4, dtype=torch.float64), torch.randn(1, 4, dtype=torch.float64)
            explicit = sum(w * expert(z, phi) for w, expert in zip(model.weights, model.experts))
            folded = model(z, phi)
            torch.testing.assert_close(explicit, folded, rtol=1e-12, atol=1e-12)
            parameters = list(model.experts.parameters())
            reference = torch.autograd.grad(explicit.square().sum(), parameters, retain_graph=True)
            actual = torch.autograd.grad(folded.square().sum(), parameters)
            for expected, observed in zip(reference, actual):
                torch.testing.assert_close(expected, observed, rtol=1e-11, atol=1e-11)

    def test_weights_frozen_and_all_experts_train(self):
        model = FixedMixture(M=6, Experts=40)
        weights = model.weights.clone()
        before = [expert.W.detach().clone() for expert in model.experts]
        x, phi = torch.randn(7, 3, 3), torch.randn(7, 3, 1)
        optimizer = torch.optim.RAdam(model.parameters(), lr=.01)
        loss = model.training_sequence(x, phi, interval=2).square().mean()
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(model.weights, weights, rtol=0, atol=0)
        for expert, original in zip(model.experts, before):
            self.assertFalse(torch.equal(expert.W, original))

    def test_one_step_alignment_and_free_rollout(self):
        model = FixedMixture(M=6)
        x0, phi = torch.randn(4, 3), torch.randn(4, 1)
        one = model(model.initial_state(x0), phi.t())[:3].t()
        rollout = model.forecast(x0, phi, 5)
        torch.testing.assert_close(one, rollout[0])
        x = x0.unsqueeze(0).expand(5, -1, -1)
        s = phi.unsqueeze(0).expand(5, -1, -1)
        torch.testing.assert_close(model.training_sequence(x, s, interval=10), rollout)
        with self.assertRaises(ValueError):
            model(model.initial_state(x0))
        autonomous = FixedMixture(M=6, phi_dim=0)
        self.assertEqual(autonomous.forecast(x0, None, 5).shape, (5, 4, 3))

    def test_sampling_balanced_and_no_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = make_source(root / 'data')
            data = Trajectories(root / 'data', split_index=60, sequence_length=7)
            for split, requested in SPLITS.items():
                x, y, phi, series, starts = data.sample_batch(split, 17, np.random.default_rng(8))
                self.assertEqual(x.shape, (7, 17, 3))
                self.assertTrue(np.all(starts + 7 < 60))
                self.assertTrue(all(any(np.isclose(v, requested)) for v in phi[0, :, 0]))
                counts = [int(np.isclose(phi[0, :, 0], v).sum()) for v in requested]
                self.assertLessEqual(max(counts) - min(counts), 1)
                for j, (s, t) in enumerate(zip(series, starts)):
                    np.testing.assert_array_equal(x[:, j], original[s, t:t + 7])
                    np.testing.assert_array_equal(y[:, j], original[s, t + 1:t + 8])
            initial, truth = data.test(20)
            np.testing.assert_array_equal(initial, original[:, 60])
            np.testing.assert_array_equal(truth, original[:, 61:81].transpose(1, 0, 2))
            del data  # release the Windows memory-map handle before temp cleanup

    def test_weight_seed_independent_of_initialization_and_split(self):
        torch.manual_seed(1)
        a = FixedMixture(M=6, weights_seed=19)
        torch.manual_seed(77)
        b = FixedMixture(M=6, weights_seed=19)
        torch.testing.assert_close(a.weights, b.weights, rtol=0, atol=0)
        self.assertFalse(torch.equal(a.experts[0].W, b.experts[0].W))

    def test_resumed_training_matches_uninterrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_source(root / 'data')
            shared = ['--source', str(root / 'data'), '--split', 'negative3', '--experts', '10',
                      '--latent-dim', '6', '--device', 'cpu', '--threads', '1', '--split-index', '60',
                      '--sequence-length', '7', '--batch-size', '4', '--epochs', '2', '--ssi', '1',
                      '--batches-per-epoch', '2']
            full = train.parse_args([*shared, '--output', str(root / 'full')])
            resumed = train.parse_args([*shared, '--output', str(root / 'resumed'), '--resume'])
            with contextlib.redirect_stdout(io.StringIO()):
                train.train(full)
                save = train.atomic_save

                def stop_after_first_epoch(path, state):
                    save(path, state)
                    if path.name == 'last_training_state.pt' and state['epoch'] == 1:
                        raise InterruptedError('simulated interruption')

                with patch.object(train, 'atomic_save', side_effect=stop_after_first_epoch):
                    with self.assertRaises(InterruptedError):
                        train.train(resumed)
                train.train(resumed)
            a, config, _ = load_run(root / 'full', 'cpu')
            b, _, _ = load_run(root / 'resumed', 'cpu')
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
            expected = FixedMixture(M=6, weights_seed=config['weights_seed']).weights
            torch.testing.assert_close(a.weights, expected, rtol=0, atol=0)
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                changed = train.parse_args([*shared, '--output', str(root / 'full'), '--weights-seed', '11'])
                train.train(changed)

    def test_scheduler_assigns_one_queue_per_gpu_and_all_splits(self):
        args = run.parse_args(['--device', 'cpu', '--gpus', '2', '4'])
        for experts in args.expert_counts:
            jobs = [(split, experts) for split in args.splits]
            self.assertEqual(len(jobs), 3)
            for slot, gpu in enumerate(args.gpus):
                for job in range(slot, len(jobs), len(args.gpus)):
                    command = run.training_command(args, *jobs[job], gpu)
                    self.assertEqual(command[command.index('--gpu-id') + 1], str(gpu))

    def test_all_finishes_ten_expert_plots_before_starting_forty(self):
        args = run.parse_args(['--device', 'cpu', '--gpus', '0', '1'])
        events = []

        def training(_args, experts, _logs, _environment):
            events.append(('train', experts))

        def evaluate(command, **_kwargs):
            counts = command[command.index('--expert-counts') + 1:command.index('--splits')]
            task = command[command.index(str(run.SCRIPT_DIR / 'evaluate.py')) + 1]
            events.append((task, tuple(map(int, counts))))
            if len(counts) == 1:
                self.assertTrue(command[command.index('--output') + 1].endswith(f'experts_{counts[0]}'))
            else:
                self.assertIn('--merge-expert-caches', command)

        with patch.object(run, 'execute_training', side_effect=training), patch.object(run.subprocess, 'run', side_effect=evaluate):
            run.execute_workflow(args, Path('unused'), {})
        self.assertEqual(events, [('train', 10), ('trajectories', (10,)), ('bifurcations', (10,)),
                                  ('train', 40), ('trajectories', (40,)), ('bifurcations', (40,)),
                                  ('bifurcations', (10, 40))])

    def test_failed_ten_expert_evaluation_prevents_forty_expert_training(self):
        args = run.parse_args(['--device', 'cpu'])
        with patch.object(run, 'execute_training') as training, patch.object(
                run.subprocess, 'run', side_effect=RuntimeError('plot failed')):
            with self.assertRaisesRegex(RuntimeError, 'plot failed'):
                run.execute_workflow(args, Path('unused'), {})
        self.assertEqual([call.args[1] for call in training.call_args_list], [10])

    def test_merge_preserves_model_order_and_rejects_inconsistent_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [{'experts': count} for count in (10, 40, 10, 40)]
            identity = {'labels': ['a10', 'a40', 'b10', 'b40'],
                        'checkpoints': ['c0', 'c1', 'c2', 'c3'],
                        'configs': ['s0', 's1', 's2', 's3'], 'seed': 8}
            grid = np.array([1.4, 1.75, 2.1])
            truth = (np.array([1.75]), np.array([2.]))
            cache_points = {}
            for count in (10, 40):
                directory = root / f'experts_{count}'
                directory.mkdir()
                (directory / 'bifurcation_cache.npz').touch()
                (directory / 'diagnostics.json').write_text('[]')
                indices = [i for i, item in enumerate(runs) if item['experts'] == count]
                cache_points[count] = {'z': (truth, [(grid, np.full(3, i)) for i in indices])}

            def matches(path, expected):
                count = int(path.parent.name.split('_')[1])
                indices = [i for i, item in enumerate(runs) if item['experts'] == count]
                self.assertEqual(expected, {key: [value[i] for i in indices] if isinstance(value, list) else value
                                            for key, value in identity.items()})
                return True

            backend = SimpleNamespace(cache_matches=Mock(side_effect=matches),
                                      load_points_cache=lambda path, count: (grid, cache_points[int(path.parent.name.split('_')[1])]))
            _, points, _ = merge_expert_caches(backend, root, runs, identity)
            self.assertEqual([int(y[0]) for x, y in points['z'][1]], [0, 1, 2, 3])
            cache_points[40]['z'] = ((truth[0], np.array([9.])), cache_points[40]['z'][1])
            with self.assertRaisesRegex(ValueError, 'different ground truth'):
                merge_expert_caches(backend, root, runs, identity)
            backend.cache_matches = lambda path, expected: False
            with self.assertRaisesRegex(ValueError, 'mismatched expert cache'):
                merge_expert_caches(backend, root, runs, identity)


if __name__ == '__main__':
    unittest.main()
