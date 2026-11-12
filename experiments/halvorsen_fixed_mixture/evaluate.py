"""Free trajectories at all nine phi values and continuous bifurcation sweeps."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEFAULT_SOURCE, SPLITS, Trajectories, configure,
                    forecast_numpy, load_run, sha256, write_json)
from dynamix.metrics.metrics import geometrical_misalignment, temporal_misalignment, MASE

MAX_ROLLOUT_ABS = 1e6  # standardized coordinates; diagnostics only, never clip raw forecasts


def rollout_status(prediction):
    values = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(values)
    valid_steps = np.all(finite & (np.abs(values) <= MAX_ROLLOUT_ABS), axis=1)
    bad = np.flatnonzero(~valid_steps)
    finite_values = np.abs(values[finite])
    return {'finite': bool(finite.all()), 'diverged': bool(len(bad)),
            'first_divergence_step': int(bad[0] + 1) if len(bad) else None,
            'max_abs_prediction': float(finite_values.max()) if finite_values.size else None}


def plotting_prediction(prediction):
    """Draw only the prefix before divergence; preserve raw arrays on disk."""
    status = rollout_status(prediction)
    end = status['first_divergence_step']
    prefix = np.asarray(prediction, dtype=np.float64)[:end - 1] if end is not None else np.asarray(prediction, dtype=np.float64)
    note = f'DIVERGED at step {end}; showing prefix' if end is not None else ''
    return prefix, note


def trajectory_limits(target, prediction):
    # Float64 is essential: even finite float32 endpoints can overflow when
    # computing their span or padded limits (the original server crash).
    combined = np.concatenate((target, prediction)).astype(np.float64)
    finite = combined[np.isfinite(combined).all(axis=1)]
    if not len(finite):
        return {}
    low, high = finite.min(axis=0), finite.max(axis=0)
    padding = np.maximum((high - low) * .03, 1e-3)
    return {name: (float(low[i] - padding[i]), float(high[i] + padding[i]))
            for i, name in enumerate(('xlim', 'ylim', 'zlim'))}


def load_backend(directory):
    """Reuse the exact ODE/standardization/peaks/plots of the existing experiments."""
    directory = Path(directory).resolve()
    script = directory / 'compare_model_bifurcation_diagrams.py'
    if not script.is_file():
        raise FileNotFoundError(f'Set ANALYSIS_DIR/--analysis-dir to diffrentbutconst_prams: {script}')
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('fixed_mixture_bifurcation_backend', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runs(args, data, device):
    runs = []
    for experts in args.expert_counts:
        for split in args.splits:
            directory = args.run_root / split / f'experts_{experts}'
            model, config, checkpoint = load_run(directory, device)
            if (config['data_identity'] != data.identity or config['split'] != split
                    or config['experts'] != experts or config['train_phi_values'] != SPLITS[split]):
                raise ValueError(f'Run does not match evaluation data or requested split: {directory}')
            runs.append({'model': model, 'config': config, 'checkpoint': checkpoint,
                         'split': split, 'experts': experts, 'label': f'{split}_experts_{experts}'})
    return runs


def measure(prediction, truth):
    row = {**rollout_status(prediction), 'dstsp': None, 'dh': None, 'mase': None, 'mse_200': None}
    pred = torch.from_numpy(np.array(prediction, dtype=np.float64))
    target = torch.from_numpy(np.array(truth, dtype=np.float64))
    results = []
    # Short-horizon errors remain meaningful if divergence occurs later.
    if not rollout_status(prediction[:200])['diverged']:
        results.append(('mse_200', torch.mean((pred[:200] - target[:200]) ** 2)))
    if not rollout_status(prediction[:10])['diverged']:
        results.append(('mase', MASE(pred, target, steps=10)))
    if not row['diverged']:
        results.append(('dstsp', geometrical_misalignment(pred, target, n_bins=30)))
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            results.append(('dh', temporal_misalignment(pred, target, smoothing=20)))
    for name, value in results:
        scalar = float(value)
        row[name] = scalar if np.isfinite(scalar) else None
    return row


def plot_trajectories(directory, prediction, truth, data, run):
    overview = plt.figure(figsize=(14, 12))
    timeseries, axes = plt.subplots(3, 3, figsize=(15, 9))
    for panel, value in enumerate(np.arange(-1, 1.01, 0.25)):
        # Same prespecified source trajectory for every run; no test-metric selection.
        index = int(np.flatnonzero(np.isclose(data.values, value, atol=1e-5, rtol=0))[0])
        seen = any(np.isclose(value, SPLITS[run['split']], atol=1e-5, rtol=0))
        title = f'phi={value:g}, a={data.raw_values[index]:.4g} ({"train" if seen else "unseen"})'
        target = np.asarray(truth[:, index], dtype=np.float64)
        pred, note = plotting_prediction(prediction[:, index])
        if note:
            title += f'\n{note}'
        ax = overview.add_subplot(3, 3, panel + 1, projection='3d')
        ax.plot(*target.T, color='black', lw=.45, alpha=.6, label='Ground truth')
        ax.plot(*pred.T, color='tab:red', lw=.45, alpha=.7, label='Fixed mixture')
        ax.set(title=title, xlabel='x', ylabel='y', zlabel='z')
        if panel == 0:
            ax.legend(fontsize=7)
        ts = axes.flat[panel]
        length = min(1000, len(pred))
        ts.plot(target[:min(1000, len(target)), 2], color='black', lw=.7, label='Ground truth')
        ts.plot(pred[:length, 2], color='tab:red', lw=.7, label='Fixed mixture')
        ts.set(title=title, xlabel='Forecast step', ylabel='z')
        if panel == 0:
            ts.legend(fontsize=7)
        comparison = plt.figure(figsize=(11, 5))
        limits = trajectory_limits(target, pred)
        for position, (trajectory, name, color) in enumerate(((target, 'Ground truth', 'black'),
                                                           (pred, 'Fixed mixture', 'tab:red')), start=1):
            ax3d = comparison.add_subplot(1, 2, position, projection='3d')
            ax3d.plot(*trajectory.T, color=color, lw=.5)
            ax3d.set(title=name, xlabel='x', ylabel='y', zlabel='z')
            ax3d.set(**limits)
            if position == 2 and note:
                ax3d.text2D(.02, .95, note, transform=ax3d.transAxes, color='tab:red', fontsize=8)
            ax3d.set_box_aspect((1, 1, 1))
        comparison.suptitle(f'{run["label"]}: {title}; source series {index}')
        comparison.tight_layout()
        tag = f'{value:+.2f}'.replace('-', 'm').replace('+', 'p').replace('.', 'p')
        comparison.savefig(directory / f'trajectory_phi_{tag}.png', dpi=150)
        plt.close(comparison)
    for figure, filename in ((overview, 'attractors_9params.png'), (timeseries, 'timeseries_9params.png')):
        figure.suptitle(run['label'])
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=150)
        plt.close(figure)


def trajectories(args, data, runs, device):
    initial, truth = data.test(args.forecast_steps)
    np.savez_compressed(args.output / 'test_reference.npz', initial_x=initial, ground_truth=truth,
                        phi=data.values, raw_parameter=data.raw_values, split_index=data.split_index)
    summary = []
    for run in runs:
        prediction = forecast_numpy(run['model'], initial, data.values, args.forecast_steps, device, args.batch_size)
        directory = args.output / run['label']
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'predictions.npz', prediction=prediction,
                            weights=run['model'].weights.cpu().numpy(), phi=data.values,
                            raw_parameter=data.raw_values, checkpoint=np.array(str(run['checkpoint'])),
                            checkpoint_sha256=np.array(sha256(run['checkpoint'])))
        rows = []
        for index, value in enumerate(data.values):
            row = {'run': run['label'], 'series': index, 'phi': float(value),
                   'raw_parameter': float(data.raw_values[index]),
                   'seen_in_training': bool(np.any(np.isclose(value, SPLITS[run['split']], atol=1e-5, rtol=0))),
                   **measure(prediction[:, index], truth[:, index])}
            rows.append(row)
        write_json(directory / 'metrics.json', rows)
        for value in np.arange(-1, 1.01, .25):
            group = [row for row in rows if np.isclose(row['phi'], value, atol=1e-5, rtol=0)]
            aggregate = {'run': run['label'], 'phi': float(value), 'seen_in_training': group[0]['seen_in_training'],
                         'n_trajectories': len(group), 'n_nonfinite': sum(not row['finite'] for row in group),
                         'n_diverged': sum(row['diverged'] for row in group)}
            for name in ('dstsp', 'dh', 'mase', 'mse_200'):
                valid = [row[name] for row in group if row[name] is not None]
                aggregate[f'median_{name}'] = float(np.median(valid)) if valid else None
            summary.append(aggregate)
        plot_trajectories(directory, prediction, truth, data, run)
        print(f'{run["label"]}: all {len(data.values)} held-out trajectories, {args.forecast_steps} free steps', flush=True)
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def merge_expert_caches(base, root, runs, identity):
    """Merge only caches matching the current checkpoints and sweep settings."""
    merged, parameters, diagnostics = {}, None, []
    for count in dict.fromkeys(run['experts'] for run in runs):
        indices = [i for i, run in enumerate(runs) if run['experts'] == count]
        directory = root / f'experts_{count}'
        cache = directory / 'bifurcation_cache.npz'
        expected = dict(identity)
        for key in ('labels', 'checkpoints', 'configs'):
            expected[key] = [identity[key][i] for i in indices]
        if not cache.is_file() or not base.cache_matches(cache, expected):
            raise ValueError(f'Cannot merge missing or mismatched expert cache: {cache}')
        grid, points = base.load_points_cache(cache, len(indices))
        if parameters is None:
            parameters = grid
        elif not np.array_equal(parameters, grid):
            raise ValueError('Expert caches have different parameter grids')
        for component, (truth, models) in points.items():
            if component not in merged:
                merged[component] = (truth, [None] * len(runs))
            reference, combined = merged[component]
            if not all(np.array_equal(a, b) for a, b in zip(reference, truth)):
                raise ValueError('Expert caches have different ground truth')
            for index, model_points in zip(indices, models):
                combined[index] = model_points
        diagnostics.extend(json.loads((directory / 'diagnostics.json').read_text(encoding='utf-8')))
    return parameters, merged, diagnostics


def bifurcations(args, data, runs, device):
    base = load_backend(args.analysis_dir)
    regime = base.regime_by_name('halvorsen')
    if args.forecast_steps <= args.discard_steps + 2:
        raise ValueError('Forecast must leave at least three steps after discard')
    if not np.isclose(data.intercept, regime.boundary, atol=1e-5):
        raise ValueError('Dataset parameter mapping does not match the Halvorsen regime')
    phi_grid = np.linspace(-1, 1, args.num_parameters)
    parameters = data.intercept + data.slope * phi_grid
    cache = args.output / 'bifurcation_cache.npz'
    identity = {
        'data': data.identity, 'labels': [run['label'] for run in runs],
        'checkpoints': [sha256(run['checkpoint']) for run in runs],
        'configs': [sha256(run['checkpoint'].parents[1] / 'config.json') for run in runs],
        'forecast_steps': args.forecast_steps, 'discard_steps': args.discard_steps,
        'num_parameters': args.num_parameters, 'rollouts': args.rollouts,
        'seed': args.seed, 'burn_in_steps': args.burn_in_steps, 'dt': args.dt,
        'batch_size': args.batch_size, 'device': str(device),
        'driver_sha256': sha256(__file__), 'helpers_sha256': sha256(Path(__file__).with_name('common.py')),
        'model_sha256': sha256(Path(__file__).resolve().parents[2] / 'src/dynamix/model/fixed_mixture.py'),
        'backend_sha256': sha256(base.__file__), 'regime_sha256': sha256(base._REGIME_MODULE.__file__),
    }
    matches = cache.is_file() and base.cache_matches(cache, identity)
    if args.plot_only and not matches:
        raise ValueError('--plot-only requires a cache matching data, checkpoints, code and all sweep settings')
    if matches and not args.force_recompute:
        parameters, points = base.load_points_cache(cache, len(runs))
        print(f'Using matching cache: {cache}', flush=True)
    elif args.merge_expert_caches is not None:
        parameters, points, diagnostics = merge_expert_caches(base, args.merge_expert_caches, runs, identity)
        base.save_points_cache(cache, identity, parameters, points)
        write_json(args.output / 'diagnostics.json', diagnostics)
        first_count = runs[0]['experts']
        shutil.copyfile(args.merge_expert_caches / f'experts_{first_count}' / 'sweep_initial_conditions.npz',
                        args.output / 'sweep_initial_conditions.npz')
        print('Merged validated expert caches; no new simulations or model rollouts', flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        initial, values, parameter_indices = [], [], []
        truth_peaks = {c: [] for c in base.COMPONENTS}
        for parameter_index, (phi, raw) in enumerate(zip(phi_grid, parameters)):
            peaks = {c: [] for c in base.COMPONENTS}
            for _ in range(args.rollouts):
                # Standardization matches the established generator/backend.
                # Ground truth is noise-free; saved test tails remain noisy.
                trajectory = base.simulate_standardized_constant_parameter(
                    regime=regime, raw_parameter=float(raw), n_steps=args.forecast_steps + 1,
                    dt=args.dt, rng=rng, rtol=1e-6, atol=1e-8,
                    max_step_multiplier=5.0, burn_in_steps=args.burn_in_steps)
                initial.append(trajectory[0])
                values.append(phi)
                parameter_indices.append(parameter_index)
                for component, index in base.COMPONENTS.items():
                    peaks[component].append(base.local_maxima(trajectory[args.discard_steps + 1:, index]))
            for component in base.COMPONENTS:
                truth_peaks[component].append(np.concatenate(peaks[component]))
            print(f'Ground truth {parameter_index + 1}/{len(parameters)}: phi={phi:.5g}, a={raw:.6g}', flush=True)
        initial = np.asarray(initial, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        parameter_indices = np.asarray(parameter_indices)
        np.savez_compressed(args.output / 'sweep_initial_conditions.npz', initial_x=initial,
                            phi=values, parameter_indices=parameter_indices, raw_parameter=parameters)
        model_peaks = {c: [[] for _ in runs] for c in base.COMPONENTS}
        diagnostics = []
        for model_index, run in enumerate(runs):
            collected = {c: [[] for _ in parameters] for c in base.COMPONENTS}
            # Batch independent parameter values; each rollout uses the SAME
            # GT initial condition across all six models, with no context.
            for start in range(0, len(initial), args.batch_size):
                end = min(len(initial), start + args.batch_size)
                prediction = forecast_numpy(run['model'], initial[start:end], values[start:end],
                                             args.forecast_steps, device, args.batch_size)
                for local, global_index in enumerate(range(start, end)):
                    parameter_index = int(parameter_indices[global_index])
                    status = rollout_status(prediction[:, local])
                    diagnostics.append({'run': run['label'], 'parameter_index': parameter_index,
                                        'phi': float(values[global_index]), 'rollout': global_index % args.rollouts,
                                        **status})
                    for component, index in base.COMPONENTS.items():
                        peaks = base.local_maxima(prediction[args.discard_steps:, local, index]) if not status['diverged'] else np.empty(0)
                        collected[component][parameter_index].append(peaks)
                print(f'{run["label"]}: bifurcation rollouts {end}/{len(initial)}', flush=True)
            for component in base.COMPONENTS:
                model_peaks[component][model_index] = [np.concatenate(peaks) for peaks in collected[component]]
        points = base.collect_points_by_component(parameters, truth_peaks, model_peaks)
        base.save_points_cache(cache, identity, parameters, points)
        write_json(args.output / 'diagnostics.json', diagnostics)
    # Combined six-model figure, per expert count, and direct 10-vs-40 per split.
    groups = {'all': list(range(len(runs)))}
    groups.update({f'experts_{count}': [i for i, run in enumerate(runs) if run['experts'] == count]
                   for count in args.expert_counts})
    groups.update({split: [i for i, run in enumerate(runs) if run['split'] == split] for split in args.splits})
    for name, indices in groups.items():
        selected = {c: (truth, [models[i] for i in indices]) for c, (truth, models) in points.items()}
        markers = [data.intercept + data.slope * np.array(SPLITS[runs[i]['split']]) for i in indices]
        base.write_plots(output=args.output / name / 'halvorsen_bifurcation.png', regime=regime,
                         parameter_values=parameters, points_by_component=selected,
                         labels=[runs[i]['label'] for i in indices], components=args.components,
                         point_size=.5, alpha=.6, dpi=150, training_parameters=markers)
    print(f'Saved bifurcations: {args.output}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('trajectories', 'bifurcations'))
    parser.add_argument('--source', type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split-index', type=int, default=80000)
    parser.add_argument('--expert-counts', type=int, choices=(10, 40), nargs='+', default=[10, 40])
    parser.add_argument('--splits', choices=SPLITS, nargs='+', default=list(SPLITS))
    parser.add_argument('--analysis-dir', type=Path, default=Path('/export/home/klenkeit/diffrentbutconst_prams'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--forecast-steps', type=int)
    parser.add_argument('--num-parameters', type=int, default=320)
    parser.add_argument('--rollouts', type=int, default=1)
    parser.add_argument('--burn-in-steps', type=int, default=5000)
    parser.add_argument('--discard-steps', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=.01)
    parser.add_argument('--components', choices=('x', 'y', 'z'), nargs='+', default=['z'])
    parser.add_argument('--merge-expert-caches', type=Path,
                        help='Build combined bifurcations from validated experts_10/experts_40 caches')
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--plot-only', action='store_true')
    cache.add_argument('--force-recompute', action='store_true')
    args = parser.parse_args(argv)
    if args.forecast_steps is None:
        args.forecast_steps = 10000 if args.task == 'trajectories' else 15000
    if (min(args.batch_size, args.rollouts, args.threads) < 1 or args.num_parameters < 2
            or args.forecast_steps < 3 or min(args.burn_in_steps, args.discard_steps) < 0
            or not 0 < args.dt < float('inf')):
        parser.error('Invalid forecast/sweep/batch/integration settings')
    if len(set(args.expert_counts)) != len(args.expert_counts) or len(set(args.splits)) != len(args.splits):
        parser.error('Expert counts and splits must be unique')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = configure(args.device, args.gpu_id, args.threads)
    data = Trajectories(args.source, args.split_index, sequence_length=1)
    runs = load_runs(args, data, device)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'evaluation_settings.json',
               {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'divergence_abs_limit': MAX_ROLLOUT_ABS})
    if args.task == 'trajectories':
        trajectories(args, data, runs, device)
    else:
        bifurcations(args, data, runs, device)


if __name__ == '__main__':
    main()
