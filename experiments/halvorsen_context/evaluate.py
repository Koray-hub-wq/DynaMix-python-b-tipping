"""Evaluate the six context-ablation runs with explicit target initial states.

Bifurcations reuse simulation, peak extraction, caching and plotting from
diffrentbutconst_prams/compare_model_bifurcation_diagrams.py. Its ordinary CLI
would replace the fixed context with a target context, so this driver supplies
the experiment-specific context and initial condition explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from dynamix.model.dynamix import DynaMix
from dynamix.model.forecaster import DynaMixForecaster
from dynamix.utilities.utilities import load_model
from dynamix.metrics.metrics import geometrical_misalignment, temporal_misalignment, MASE


def load_runs(args, manifest, device):
    runs = []
    for mode in ('fixed', 'matched'):
        for split in manifest['splits']:
            directory = args.run_root / split / mode
            config = json.loads((directory / 'config.json').read_text(encoding='utf-8-sig'))
            model = DynaMix(M=config['latent_dim'], N=3, Experts=config['experts'],
                            P=config['pwl_units'], hidden_dim=config.get('hidden_dim', 50),
                            expert_type=config['expert_type'], phi_dim=config['phi_dim'],
                            probabilistic_expert=config.get('probabilistic_expert', False)).to(device)
            checkpoint = directory / 'checkpoints/final_model.safetensors'
            load_model(checkpoint, model, device=str(device))
            model.eval()
            runs.append({'split': split, 'mode': mode, 'label': f'{split}_{mode}',
                         'model': model, 'forecaster': DynaMixForecaster(model),
                         'checkpoint': checkpoint})
    return runs


def context_mode(args, run):
    return run['mode'] if args.eval_context == 'trained' else args.eval_context


@torch.no_grad()
def forecast(run, context, initial_x, phi_values, horizon, device, seed):
    """Free rollout. Keep a mean of expert weights to aid interpretation."""
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    # Read-only memory maps must not be shared with writable Torch tensors.
    context = torch.tensor(context, dtype=torch.float32, device=device)
    initial_x = torch.tensor(initial_x, dtype=torch.float32, device=device)
    phi = torch.tensor(phi_values, dtype=torch.float32, device=device).reshape(1, -1, 1)
    phi = phi.expand(horizon, -1, -1)
    total = torch.zeros(run['model'].Experts, initial_x.shape[0], device=device)

    def collect_weights(_module, _inputs, weights):
        total.add_(weights.detach())

    hook = run['model'].gating_network.register_forward_hook(collect_weights)
    try:
        prediction = run['forecaster'].forecast(context, horizon, initial_x=initial_x,
                                               phi_future=phi, standardize=False)
    finally:
        hook.remove()
    return prediction.cpu().numpy(), (total / horizon).t().cpu().numpy()


def trajectory_plots(directory, prediction, truth, values, raw_values, rows, label):
    """Show one median-geometrical-error trajectory per phi; cache contains all."""
    fig3d = plt.figure(figsize=(13, 12))
    figts, axes = plt.subplots(3, 3, figsize=(15, 9), squeeze=False)
    for panel, value in enumerate(np.unique(values)):
        indices = np.flatnonzero(np.isclose(values, value, atol=1e-5, rtol=0))
        valid = [int(i) for i in indices if rows[int(i)]['dstsp'] is not None]
        ranked = sorted(valid, key=lambda i: rows[i]['dstsp'])
        index = ranked[len(ranked) // 2] if ranked else int(indices[0])
        ax = fig3d.add_subplot(3, 3, panel + 1, projection='3d')
        target, pred = truth[:, index], prediction[:, index]
        ax.plot(*target.T, color='black', lw=0.4, alpha=0.6, label='Ground truth')
        ax.plot(*pred.T, color='tab:red', lw=0.4, alpha=0.7, label='DynaMix')
        title = f'phi={value:g}, a={raw_values[index]:.4g}, series={index}'
        ax.set_title(title, fontsize=9)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        ts = axes.flat[panel]
        steps = min(1000, len(pred))
        ts.plot(target[:steps, 2], color='black', lw=0.8, label='Ground truth')
        ts.plot(pred[:steps, 2], color='tab:red', lw=0.8, label='DynaMix')
        ts.set_title(title, fontsize=9)
        ts.set(xlabel='Prediction step', ylabel='z')
    axes.flat[0].legend()
    for fig, filename in ((fig3d, 'attractors_9params.png'), (figts, 'timeseries_9params.png')):
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)


def trajectories(args, manifest, runs, device):
    evaluation = args.data_root / 'evaluation'
    test = np.load(evaluation / 'test.npy', mmap_mode='r')
    phi = np.load(evaluation / 'test_phi.npy', mmap_mode='r')
    raw = np.load(evaluation / 'test_phi_raw.npy', mmap_mode='r')
    initial_index = manifest['test_initial_index']
    available = len(test) - initial_index - 1
    horizon = available if args.forecast_steps == 0 else args.forecast_steps
    if not 20 < horizon <= available:
        raise ValueError(f'Trajectory forecast needs 21..{available} steps')
    initial = test[initial_index]
    truth = test[initial_index + 1:initial_index + 1 + horizon]
    values, raw_values = phi[initial_index, :, 0], raw[initial_index, :, 0]
    np.savez_compressed(args.output / 'test_reference.npz', ground_truth=truth, initial_x=initial,
                        phi=values, raw_parameter=raw_values)
    summary = []
    for run in runs:
        mode = context_mode(args, run)
        contexts = np.load(evaluation / f'{mode}_context.npy', mmap_mode='r')
        prediction = np.empty_like(truth)
        weights = np.empty((test.shape[1], run['model'].Experts), dtype=np.float32)
        for start in range(0, test.shape[1], args.batch_size):
            end = min(test.shape[1], start + args.batch_size)
            prediction[:, start:end], weights[start:end] = forecast(
                run, contexts[:, start:end], initial[start:end], values[start:end], horizon,
                device, args.seed + start)
        directory = args.output / run['label']
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / 'predictions.npz', prediction=prediction,
                            mean_expert_weights=weights, phi=values, raw_parameter=raw_values,
                            context_mode=np.array(mode), checkpoint=np.array(str(run['checkpoint'])))
        rows = []
        for index, value in enumerate(values):
            row = {'run': run['label'], 'context': mode, 'series': index,
                   'phi': float(value), 'raw_parameter': float(raw_values[index]),
                   'seen_in_training': bool(np.any(np.isclose(value, manifest['splits'][run['split']], atol=1e-5))),
                   'finite': bool(np.isfinite(prediction[:, index]).all()),
                   'dstsp': None, 'dh': None, 'mase': None, 'mse_200': None}
            if row['finite']:
                pred = torch.from_numpy(prediction[:, index].copy())
                target = torch.from_numpy(truth[:, index].copy())
                for name, result in (
                    ('dstsp', geometrical_misalignment(pred, target, n_bins=30)),
                    ('dh', temporal_misalignment(pred, target, smoothing=20)),
                    ('mase', MASE(target, pred, steps=10)),
                    ('mse_200', torch.mean((pred[:200] - target[:200]) ** 2)),
                ):
                    scalar = float(result)
                    row[name] = scalar if np.isfinite(scalar) else None
            rows.append(row)
        (directory / 'metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
        trajectory_plots(directory, prediction, truth, values, raw_values, rows, run['label'])
        for value in np.unique(values):
            group = [row for row in rows if np.isclose(row['phi'], value, atol=1e-5)]
            aggregate = {'run': run['label'], 'context': mode, 'phi': float(value),
                         'n_trajectories': len(group), 'n_diverged': sum(not row['finite'] for row in group),
                         'seen_in_training': group[0]['seen_in_training']}
            for metric in ('dstsp', 'dh', 'mase', 'mse_200'):
                valid = [row[metric] for row in group if row[metric] is not None]
                aggregate[f'median_{metric}'] = float(np.median(valid)) if valid else None
            summary.append(aggregate)
        print(f'{run["label"]}: saved {test.shape[1]} test rollouts ({horizon} steps), context={mode}', flush=True)
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def bifurcations(args, manifest, runs, device):
    analysis_dir = args.analysis_dir.resolve()
    if not (analysis_dir / 'compare_model_bifurcation_diagrams.py').is_file():
        raise FileNotFoundError(f'Set ANALYSIS_DIR to diffrentbutconst_prams: {analysis_dir}')
    sys.path.insert(0, str(analysis_dir))
    base = importlib.import_module('compare_model_bifurcation_diagrams')
    regime = base.regime_by_name('halvorsen')
    context_length = manifest['context_length']
    if args.forecast_steps <= args.discard_steps + 2 or args.num_parameters < 2 or args.rollouts < 1:
        raise ValueError('Invalid sweep size, rollout count or discard/forecast length')
    reference = np.load(args.data_root / 'evaluation/reference_context_pool.npy')
    # Sweep the normalized range represented in the source data, using its exact raw mapping.
    phi_values = np.array(manifest['phi_values'])
    raw_values = np.array(manifest['raw_values'])
    slope, intercept = np.polyfit(phi_values, raw_values, 1)
    if not np.allclose(raw_values, slope * phi_values + intercept, atol=1e-5, rtol=0):
        raise ValueError('Source data does not have an affine phi-to-raw mapping')
    parameters = np.linspace(intercept - slope, intercept + slope, args.num_parameters)
    cache = args.output / 'bifurcation_cache.npz'
    identity = {
        'experiment_manifest_sha256': hashlib.sha256((args.data_root / 'manifest.json').read_bytes()).hexdigest(),
        'checkpoints': [hashlib.sha256(run['checkpoint'].read_bytes()).hexdigest() for run in runs],
        'labels': [run['label'] for run in runs], 'eval_context': args.eval_context,
        'forecast_steps': args.forecast_steps, 'discard_steps': args.discard_steps,
        'num_parameters': args.num_parameters, 'rollouts': args.rollouts,
        'seed': args.seed, 'burn_in_steps': args.burn_in_steps, 'dt': args.dt,
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'simulation_script_sha256': hashlib.sha256((analysis_dir / 'compare_model_bifurcation_diagrams.py').read_bytes()).hexdigest(),
        'regime_definitions_sha256': hashlib.sha256(Path(base._REGIME_MODULE.__file__).read_bytes()).hexdigest(),
    }
    if args.plot_only and not base.cache_matches(cache, identity):
        raise ValueError('--plot-only requires a cache matching this experiment and all sweep settings')
    if cache.is_file() and base.cache_matches(cache, identity) and not args.force_recompute:
        parameters, points = base.load_points_cache(cache, len(runs))
        print(f'Using cache {cache}', flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        truth_peaks = {c: [] for c in base.COMPONENTS}
        model_peaks = {c: [[] for _ in runs] for c in base.COMPONENTS}
        diagnostics = []
        for parameter_index, raw_parameter in enumerate(parameters):
            trajectory = base.simulate_standardized_constant_parameter(
                regime=regime, raw_parameter=float(raw_parameter),
                n_steps=context_length + args.forecast_steps + 1, dt=args.dt, rng=rng,
                rtol=1e-6, atol=1e-8, max_step_multiplier=5.0, burn_in_steps=args.burn_in_steps)
            truth = trajectory[context_length + 1 + args.discard_steps:]
            value = (raw_parameter - intercept) / slope
            for component, index in base.COMPONENTS.items():
                truth_peaks[component].append(base.local_maxima(truth[:, index]))
            for model_index, run in enumerate(runs):
                peaks = {c: [] for c in base.COMPONENTS}
                for rollout in range(args.rollouts):
                    mode = context_mode(args, run)
                    context = (reference[:, rollout % reference.shape[1]:rollout % reference.shape[1] + 1]
                               if mode == 'fixed' else trajectory[:context_length, None])
                    prediction, weights = forecast(run, context, trajectory[context_length:context_length + 1],
                                                   [value], args.forecast_steps, device,
                                                   args.seed + parameter_index * args.rollouts + rollout)
                    pred = prediction[args.discard_steps:, 0]
                    finite = bool(np.isfinite(prediction).all())
                    diagnostics.append({'run': run['label'], 'raw_parameter': float(raw_parameter),
                                        'phi': float(value), 'rollout': rollout, 'finite': finite,
                                        'mean_expert_weights': [float(w) if np.isfinite(w) else None for w in weights[0]]})
                    for component, index in base.COMPONENTS.items():
                        peaks[component].append(base.local_maxima(pred[:, index]) if finite else np.empty(0))
                for component in base.COMPONENTS:
                    model_peaks[component][model_index].append(np.concatenate(peaks[component]))
            print(f'Bifurcation {parameter_index + 1}/{len(parameters)}: a={raw_parameter:.6g}, phi={value:.6g}', flush=True)
        points = base.collect_points_by_component(parameters, truth_peaks, model_peaks)
        base.save_points_cache(cache, identity, parameters, points)
        (args.output / 'diagnostics.json').write_text(json.dumps(diagnostics, allow_nan=False), encoding='utf-8')
    # One comparison for all six runs, plus one for each context condition.
    for group in ('all', 'fixed', 'matched'):
        indices = [i for i, run in enumerate(runs) if group == 'all' or run['mode'] == group]
        selected = {c: (truth, [models[i] for i in indices]) for c, (truth, models) in points.items()}
        markers = [intercept + slope * np.array(manifest['splits'][runs[i]['split']]) for i in indices]
        base.write_plots(output=args.output / f'{group}/halvorsen_bifurcation.png', regime=regime,
                         parameter_values=parameters, points_by_component=selected,
                         labels=[runs[i]['label'] for i in indices], components=args.components,
                         point_size=0.5, alpha=0.6, dpi=150, training_parameters=markers)
    print(f'Saved bifurcation plots and cache: {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('trajectories', 'bifurcations'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, default=Path('/export/home/klenkeit/diffrentbutconst_prams'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--eval-context', choices=('trained', 'fixed', 'matched'), default='trained')
    parser.add_argument('--forecast-steps', type=int, default=0)
    parser.add_argument('--num-parameters', type=int, default=320)
    parser.add_argument('--rollouts', type=int, default=1)
    parser.add_argument('--burn-in-steps', type=int, default=5000)
    parser.add_argument('--discard-steps', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--components', nargs='+', choices=('x', 'y', 'z'), default=['z'])
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument('--plot-only', action='store_true')
    cache_mode.add_argument('--force-recompute', action='store_true')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0 or args.dt <= 0 or args.discard_steps < 0 or args.burn_in_steps < 0:
        parser.error('Invalid batch size, thread count, integration step or burn-in/discard length')
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        if not torch.cuda.is_available() or not 0 <= args.gpu_id < torch.cuda.device_count():
            raise RuntimeError('Requested CUDA GPU is unavailable')
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if args.device == 'cuda' else 'cpu')
    manifest = json.loads((args.data_root / 'manifest.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args, manifest, device)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / 'evaluation_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    if args.task == 'trajectories':
        trajectories(args, manifest, runs, device)
    else:
        bifurcations(args, manifest, runs, device)


if __name__ == '__main__':
    main()
