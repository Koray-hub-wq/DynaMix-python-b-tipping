"""Full-trajectory sampling and checkpoint helpers for the fixed-mixture ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from dynamix.model.fixed_mixture import FixedMixture

SPLITS = {
    'negative3': [-1.0, -0.75, -0.5],
    'spanning5': [-1.0, -0.5, 0.0, 0.5, 1.0],
    'interleaved3': [-0.25, 0.25, 0.75],
}
DEFAULT_SOURCE = '/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def configure(device='cuda', gpu_id=0, threads=4):
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    if device == 'cuda':
        if not torch.cuda.is_available() or not 0 <= gpu_id < torch.cuda.device_count():
            raise RuntimeError(f'CUDA GPU {gpu_id} is unavailable')
        torch.cuda.set_device(gpu_id)
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


class Trajectories:
    """Memory map full trajectories; sample only before the chronological split.

    phi is a data selector and a known external input, never a context. Sampling
    and its RNG are independent of model size/initialization and GPU RNG state.
    """

    def __init__(self, source, split_index=80000, sequence_length=200):
        self.source = Path(source).resolve()
        if not (self.source / 'trajectories.npy').is_file():
            self.source = self.source / 'trajectories'
        self.x = np.load(self.source / 'trajectories.npy', mmap_mode='r')
        if self.x.ndim != 3 or self.x.shape[2] != 3:
            raise ValueError(f'Need trajectories (series,time,3), got {self.x.shape}')
        if not 1 <= sequence_length < split_index < self.x.shape[1] - 1:
            raise ValueError('Require 1<=sequence_length<split_index<trajectory_length-1')
        self.split_index, self.sequence_length = split_index, sequence_length
        self.values = self._constants('trajectory_phi')
        self.raw_values = self._constants('trajectory_phi_raw')
        expected = np.arange(-1, 1.01, 0.25)
        if (any(not np.any(np.isclose(self.values, v, atol=1e-5, rtol=0)) for v in expected)
                or any(not np.any(np.isclose(v, expected, atol=1e-5, rtol=0)) for v in self.values)):
            raise ValueError(f'Need exactly the nine phi values [-1:0.25:1]; found {np.unique(self.values)}')
        self.slope, self.intercept = np.polyfit(self.values, self.raw_values, 1)
        if self.slope <= 0 or not np.allclose(self.raw_values, self.slope * self.values + self.intercept, atol=1e-5, rtol=0):
            raise ValueError('Need an increasing affine phi-to-raw mapping')
        metadata = self.source / 'metadata.json'
        self.metadata = json.loads(metadata.read_text(encoding='utf-8-sig')) if metadata.is_file() else {}
        # Hash observations and parameter files so moved/replaced data cannot be
        # silently associated with an old checkpoint or bifurcation cache.
        files = ['trajectories.npy', 'trajectory_phi.npy', 'trajectory_phi_raw.npy']
        files += [name + '.npy' for name in ('trajectory_phi_constants', 'trajectory_phi_raw_constants')
                  if (self.source / (name + '.npy')).is_file()]
        if metadata.is_file():
            files.append('metadata.json')
        self.identity = {'sha256': {name: sha256(self.source / name) for name in files},
                         'shape': list(self.x.shape), 'split_index': split_index}

    def _constants(self, name):
        array = np.load(self.source / f'{name}.npy', mmap_mode='r')
        if array.shape != (*self.x.shape[:2], 1):
            raise ValueError(f'{name} must have shape {(*self.x.shape[:2], 1)}')
        values = np.array(array[:, 0, 0], dtype=np.float64)
        for i, value in enumerate(values):
            if not np.isfinite(value) or not np.allclose(array[i, :, 0], value, atol=1e-5, rtol=0):
                raise ValueError(f'{name}: series {i} must be constant and finite')
        constants = self.source / f'{name}_constants.npy'
        if constants.is_file() and not np.allclose(np.load(constants).reshape(-1), values, atol=1e-5, rtol=0):
            raise ValueError(f'{constants} disagrees with {name}.npy')
        return values

    def sample_batch(self, split, batch_size, rng):
        requested = np.asarray(SPLITS[split])
        # Counts differ by at most one, including batch_size < number of phis.
        rng.shuffle(requested)
        balanced = np.tile(requested, (batch_size + len(requested) - 1) // len(requested))[:batch_size]
        rng.shuffle(balanced)
        series = np.array([rng.choice(np.flatnonzero(np.isclose(self.values, value, atol=1e-5, rtol=0)))
                           for value in balanced])
        starts = rng.integers(0, self.split_index - self.sequence_length, size=batch_size)
        windows = np.stack([self.x[s, t:t + self.sequence_length + 1]
                            for s, t in zip(series, starts)], axis=1).astype(np.float32)
        if not np.isfinite(windows).all():
            raise ValueError('Non-finite observations in training window')
        phi = np.broadcast_to(self.values[series][None, :, None], (self.sequence_length, batch_size, 1)).copy().astype(np.float32)
        return windows[:-1].copy(), windows[1:].copy(), phi, series, starts

    def test(self, horizon):
        if not 1 <= horizon <= self.x.shape[1] - self.split_index - 1:
            raise ValueError('Test horizon exceeds held-out portion of trajectory')
        initial = np.array(self.x[:, self.split_index], dtype=np.float32)
        truth = np.array(self.x[:, self.split_index + 1:self.split_index + horizon + 1].transpose(1, 0, 2), dtype=np.float32)
        if not np.isfinite(initial).all() or not np.isfinite(truth).all():
            raise ValueError('Non-finite observations in test data')
        return initial, truth


def load_run(directory, device):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    model = FixedMixture(M=config['latent_dim'], Experts=config['experts'], P=config['pwl_units'],
                         phi_dim=config['phi_dim'], weights_seed=config['weights_seed']).to(device)
    checkpoint = directory / 'checkpoints/final_model.pt'
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    saved_weights = np.asarray(config['weights'], dtype=np.float32)
    if not np.array_equal(model.weights.cpu().numpy(), saved_weights):
        raise ValueError(f'Checkpoint/config weights disagree: {directory}')
    model.eval()
    return model, config, checkpoint


def forecast_numpy(model, initial, values, horizon, device, batch_size=16):
    prediction = np.empty((horizon, len(initial), 3), dtype=np.float32)
    for start in range(0, len(initial), batch_size):
        end = min(len(initial), start + batch_size)
        x0 = torch.tensor(initial[start:end], device=device)
        phi = torch.tensor(np.asarray(values[start:end], dtype=np.float32)[:, None], device=device) if model.phi_dim else None
        prediction[:, start:end] = model.forecast(x0, phi, horizon).cpu().numpy()
    return prediction
