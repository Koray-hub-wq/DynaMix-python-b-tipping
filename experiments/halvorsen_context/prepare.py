"""Build paired fixed/matched-context datasets from the nine-parameter trajectories.

The targets and test tails are identical across conditions. Full trajectories
are required so zero overlap and a held-out test tail can be guaranteed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPLITS = {
    'negative3': [-1.0, -0.75, -0.5],
    'interleaved3': [-0.25, 0.25, 0.75],
    'spanning5': [-1.0, -0.5, 0.0, 0.5, 1.0],
}


def trajectory_dir(source: Path) -> Path:
    for candidate in (source, source / 'trajectories', source.parent / 'trajectories'):
        if (candidate / 'trajectories.npy').is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f'No full trajectories found at {source}. Set DATA_SOURCE to the directory '
        'containing trajectories.npy, trajectory_phi.npy and trajectory_phi_raw.npy. '
        'Existing overlapping training windows cannot guarantee the requested split.'
    )


def constant_values(array: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if array.shape != (*shape, 1):
        raise ValueError(f'{name}: expected {(*shape, 1)}, got {array.shape}')
    constants = np.asarray(array[:, 0, 0])
    for i, value in enumerate(constants):
        if not np.isfinite(value) or not np.allclose(array[i, :, 0], value, atol=1e-5, rtol=0):
            raise ValueError(f'{name}: trajectory {i} is not constant and finite')
    return constants


def reference_window(rng, pool, target_series, target_start, target_end, prefix_end, length):
    """Sample a reference context, excluding overlap with the target interval."""
    other = pool[pool != target_series]
    if len(other):
        series = int(rng.choice(other))
        return series, int(rng.integers(prefix_end - length + 1))
    series = int(pool[0])
    if series != target_series:
        return series, int(rng.integers(prefix_end - length + 1))
    # One reference trajectory: sample only intervals before or after the target.
    before = max(0, min(prefix_end - length, target_start - length) + 1)
    after = max(0, prefix_end - length - target_end + 1)
    if before + after == 0:
        raise ValueError('No non-overlapping reference context remains before the test tail')
    draw = int(rng.integers(before + after))
    return series, draw if draw < before else target_end + draw - before


def prepare(source: Path, output: Path, *, reference_phi=-0.75, context_length=500,
            prediction_length=200, test_length=10000, windows_per_trajectory=16, seed=8):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; use a new DATA_ROOT for a new experiment')
    source = trajectory_dir(source)
    x = np.load(source / 'trajectories.npy', mmap_mode='r')
    phi = np.load(source / 'trajectory_phi.npy', mmap_mode='r')
    raw = np.load(source / 'trajectory_phi_raw.npy', mmap_mode='r')
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f'Expected Halvorsen trajectories (series, time, 3), got {x.shape}')
    if context_length < 2 or min(prediction_length, windows_per_trajectory) <= 0:
        raise ValueError('Invalid context, prediction length or window count')
    if test_length <= context_length + 1:
        raise ValueError('test_length must leave points after context and initialization')
    values = constant_values(phi, x.shape[:2], 'trajectory_phi')
    raw_values = constant_values(raw, x.shape[:2], 'trajectory_phi_raw')
    expected = np.arange(-1, 1.01, 0.25)
    if any(not np.any(np.isclose(values, v, atol=1e-5, rtol=0)) for v in expected):
        raise ValueError(f'Need all nine phi values {expected.tolist()}; found {np.unique(values)}')
    pool = np.flatnonzero(np.isclose(values, reference_phi, atol=1e-5, rtol=0))
    if not len(pool):
        raise ValueError(f'No reference trajectories at phi={reference_phi}')
    n_series, n_steps, _ = x.shape
    prefix_end = n_steps - test_length
    data_length = prediction_length + 1  # x0 plus exactly prediction_length transitions
    max_start = prefix_end - context_length - data_length
    if max_start < 0:
        raise ValueError('Trajectories are too short for disjoint training and test segments')
    metadata_path = source / 'metadata.json'
    source_metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig')) if metadata_path.exists() else {}
    rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed + 1)
    series = np.repeat(np.arange(n_series), windows_per_trajectory)
    starts = rng.integers(0, max_start + 1, size=len(series))
    data_starts = starts + context_length
    targets = np.stack([x[s, t:t + data_length] for s, t in zip(series, data_starts)], axis=1).astype(np.float32)
    matched = np.stack([x[s, t:t + context_length] for s, t in zip(series, starts)], axis=1).astype(np.float32)
    refs = [reference_window(reference_rng, pool, s, t, t + data_length, prefix_end, context_length)
            for s, t in zip(series, data_starts)]
    ref_series, ref_starts = np.array(refs, dtype=np.int64).T
    fixed = np.stack([x[s, t:t + context_length] for s, t in refs], axis=1).astype(np.float32)
    test = np.asarray(x[:, -test_length:].transpose(1, 0, 2), dtype=np.float32)
    if not all(np.isfinite(a).all() for a in (targets, matched, fixed, test)):
        raise ValueError('Non-finite observations in sampled data')
    test_refs = np.array([reference_rng.choice(pool[pool != s] if np.any(pool != s) else pool)
                          for s in range(n_series)])
    output.mkdir(parents=True, exist_ok=True)
    evaluation = output / 'evaluation'
    evaluation.mkdir()
    eval_arrays = {
        'test': test,
        'test_phi': np.asarray(phi[:, -test_length:].transpose(1, 0, 2), dtype=np.float32),
        'test_phi_raw': np.asarray(raw[:, -test_length:].transpose(1, 0, 2), dtype=np.float32),
        'matched_context': test[:context_length],
        'fixed_context': test[:context_length, test_refs],
        'reference_context_pool': test[:context_length, pool],
        'fixed_context_series_indices': test_refs,
        'reference_series_indices': pool,
    }
    for name, array in eval_arrays.items():
        np.save(evaluation / f'{name}.npy', array)
    manifest = {
        'source_dir': str(source), 'reference_phi': reference_phi,
        'context_length': context_length, 'prediction_length': prediction_length,
        'data_length': data_length, 'overlap': 0, 'test_length': test_length,
        'test_initial_index': context_length, 'windows_per_trajectory': windows_per_trajectory,
        'seed': seed, 'evaluation_dir': str(evaluation), 'splits': SPLITS,
        'phi_values': [float(v) for v in values], 'raw_values': [float(v) for v in raw_values],
        'source_metadata': source_metadata,
        'layout': 'Context [s,s+C), x0 at s+C; score x[s+C+1:s+C+H+1]. Test tail is excluded from all training inputs.',
        'runs': [],
    }
    for split, requested in SPLITS.items():
        keep = np.any(np.isclose(values[series, None], requested, atol=1e-5, rtol=0), axis=1)
        for mode, contexts, context_series, context_starts in (
            ('matched', matched, series, starts), ('fixed', fixed, ref_series, ref_starts)
        ):
            directory = output / split / mode
            directory.mkdir(parents=True)
            count = int(keep.sum())
            arrays = {
                'data': targets[:, keep], 'context': contexts[:, keep],
                'phi': np.broadcast_to(values[series[keep]][None, :, None], (data_length, count, 1)),
                'phi_raw': np.broadcast_to(raw_values[series[keep]][None, :, None], (data_length, count, 1)),
                'context_phi': np.broadcast_to(values[context_series[keep]][None, :, None], (context_length, count, 1)),
                'context_phi_raw': np.broadcast_to(raw_values[context_series[keep]][None, :, None], (context_length, count, 1)),
                'window_series_indices': series[keep], 'start_indices': starts[keep],
                'data_start_indices': data_starts[keep], 'context_series_indices': context_series[keep],
                'context_start_indices': context_starts[keep],
            }
            for name, array in arrays.items():
                np.save(directory / f'{name}.npy', array)
            details = {'mode': mode, 'selected_phi_values': requested,
                       'reference_phi': reference_phi, 'source_dir': str(source), 'overlap': 0,
                       'context_length': context_length, 'prediction_length': prediction_length,
                       'regimes': source_metadata.get('regimes', []),
                       'parameter_filter': {'selected_phi_values': requested,
                                            'selected_raw_parameter_values': [float(v) for v in np.unique(raw_values[series[keep]])]}}
            (directory / 'metadata.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
            manifest['runs'].append({'split': split, 'mode': mode, 'data_dir': str(directory)})
            print(f'{split}/{mode}: {count} windows; context={context_length}, predictions={prediction_length}')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-phi', type=float, default=-0.75)
    parser.add_argument('--context-length', type=int, default=500)
    parser.add_argument('--prediction-length', type=int, default=200)
    parser.add_argument('--test-length', type=int, default=10000)
    parser.add_argument('--windows-per-trajectory', type=int, default=16)
    parser.add_argument('--seed', type=int, default=8)
    args = parser.parse_args()
    prepare(args.source, args.output, reference_phi=args.reference_phi,
            context_length=args.context_length, prediction_length=args.prediction_length,
            test_length=args.test_length, windows_per_trajectory=args.windows_per_trajectory, seed=args.seed)


if __name__ == '__main__':
    main()
