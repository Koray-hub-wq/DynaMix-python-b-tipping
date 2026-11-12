"""Complete training and plots for 10 experts before starting 40 experts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import threading

from common import DEFAULT_SOURCE, REPO_ROOT, SPLITS, Trajectories, configure, write_json

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('all', 'train', 'trajectories', 'bifurcations'), default='all')
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('DATA_SOURCE', DEFAULT_SOURCE)))
    parser.add_argument('--output-root', type=Path, default=os.environ.get('EXPERIMENT_ROOT'))
    parser.add_argument('--analysis-dir', type=Path, default=Path(os.environ.get('ANALYSIS_DIR', '/export/home/klenkeit/diffrentbutconst_prams')))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=os.environ.get('DEVICE', 'cuda'))
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--expert-counts', type=int, choices=(10, 40), nargs='+', default=[10, 40])
    parser.add_argument('--splits', choices=SPLITS, nargs='+', default=list(SPLITS))
    for name, default in {'seed': 8, 'weights-seed': 8, 'latent-dim': 30, 'phi-dim': 1,
                          'threads': 4, 'split-index': 80000, 'sequence-length': 200,
                          'epochs': 500, 'batch-size': 8, 'batches-per-epoch': 20,
                          'teacher-forcing-interval': 10, 'ssi': 25,
                          'trajectory-steps': 10000, 'bifurcation-steps': 15000,
                          'num-parameters': 320, 'rollouts': 1, 'burn-in-steps': 5000,
                          'discard-steps': 1000, 'eval-batch-size': 16}.items():
        parser.add_argument(f'--{name}', type=int, default=int(os.environ.get(name.upper().replace('-', '_'), default)))
    for name, default in {'alpha': 1.0, 'start-lr': .004, 'end-lr': .00001,
                          'noise-level': .02, 'dt': .01}.items():
        parser.add_argument(f'--{name}', type=float, default=float(os.environ.get(name.upper().replace('-', '_'), default)))
    parser.add_argument('--components', choices=('x', 'y', 'z'), nargs='+', default=['z'])
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--plot-only', action='store_true')
    cache.add_argument('--force-recompute', action='store_true')
    args = parser.parse_args(argv)
    if args.gpus is None:
        try:
            args.gpus = [int(value) for value in os.environ.get('GPUS', os.environ.get('GPU_ID', '0')).replace(',', ' ').split()]
        except ValueError:
            parser.error('GPUS must contain integer GPU IDs')
    if not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error('GPU IDs must be unique nonnegative integers')
    if len(set(args.expert_counts)) != len(args.expert_counts) or len(set(args.splits)) != len(args.splits):
        parser.error('Expert counts and splits must be unique')
    if args.phi_dim not in (0, 1) or args.latent_dim < 3:
        parser.error('Need phi-dim 0/1 and latent-dim >= 3')
    if min(args.epochs, args.batch_size, args.batches_per_epoch, args.ssi, args.sequence_length,
           args.teacher_forcing_interval, args.threads, args.eval_batch_size, args.rollouts) < 1:
        parser.error('Training/batch/rollout counts must be positive')
    if (args.num_parameters < 2 or args.trajectory_steps < 3 or args.bifurcation_steps <= args.discard_steps + 2
            or min(args.discard_steps, args.burn_in_steps) < 0 or not 0 < args.dt < float('inf')):
        parser.error('Invalid evaluation/sweep/integration settings')
    if (not 0 <= args.alpha <= 1 or not 0 <= args.noise_level < float('inf')
            or not 0 < args.end_lr <= args.start_lr < float('inf')):
        parser.error('Invalid teacher forcing strength, noise or learning rate')
    if args.output_root is None:
        args.output_root = REPO_ROOT / 'results/halvorsen_fixed_mixture' / f'seed_{args.seed}_w{args.weights_seed}_m{args.latent_dim}_p2_phi{args.phi_dim}'
    args.output_root = args.output_root.resolve()
    return args


def training_command(args, split, experts, gpu):
    command = [sys.executable, '-u', str(SCRIPT_DIR / 'train.py'), '--source', str(args.source.resolve()),
               '--output', str(args.output_root / 'models' / split / f'experts_{experts}'),
               '--split', split, '--experts', str(experts), '--device', args.device,
               '--gpu-id', str(gpu), '--resume']
    for name in ('latent_dim', 'phi_dim', 'weights_seed', 'seed', 'split_index', 'sequence_length',
                 'epochs', 'batch_size', 'batches_per_epoch', 'teacher_forcing_interval', 'alpha',
                 'start_lr', 'end_lr', 'noise_level', 'ssi', 'threads'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    return command


def evaluation_command(args, task, experts=None):
    output = args.output_root / task
    if experts is not None:
        output = output / f'experts_{experts}'
    command = [sys.executable, '-u', str(SCRIPT_DIR / 'evaluate.py'), task,
               '--source', str(args.source.resolve()), '--run-root', str(args.output_root / 'models'),
               '--output', str(output), '--split-index', str(args.split_index),
               '--device', args.device, '--gpu-id', str(args.gpus[0]), '--threads', str(args.threads),
               '--seed', str(args.seed), '--batch-size', str(args.eval_batch_size),
               '--expert-counts', *map(str, args.expert_counts if experts is None else [experts]), '--splits', *args.splits,
               '--forecast-steps', str(args.trajectory_steps if task == 'trajectories' else args.bifurcation_steps)]
    if task == 'bifurcations':
        command.extend(['--analysis-dir', str(args.analysis_dir.resolve()), '--components', *args.components])
        for name in ('num_parameters', 'rollouts', 'burn_in_steps', 'discard_steps', 'dt'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        if args.plot_only:
            command.append('--plot-only')
        if args.force_recompute:
            command.append('--force-recompute')
    return command


def execute_training(args, experts, logs, environment):
    tasks = [(split, experts) for split in args.splits]
    commands = [training_command(args, split, experts, args.gpus[i % len(args.gpus)])
                for i, (split, experts) in enumerate(tasks)]
    children, lock = set(), threading.Lock()
    stopping = threading.Event()

    def worker(slot):
        for job in range(slot, len(tasks), len(args.gpus)):
            if stopping.is_set():
                return
            split, count = tasks[job]
            log = logs / f'{split}_experts_{count}.log'
            print(f'Train {split}/experts_{count} on {args.device}:{args.gpus[slot]}; log: {log}', flush=True)
            with log.open('a', encoding='utf-8') as stream:
                with lock:
                    if stopping.is_set():
                        return
                    process = subprocess.Popen(commands[job], env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    children.add(process)
                try:
                    code = process.wait()
                finally:
                    with lock:
                        children.discard(process)
                if code:
                    raise RuntimeError(f'Training failed (exit {code}); inspect {log}')

    # This pool finishes before any plots or another expert count may start.
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, slot) for slot in range(min(len(tasks), len(args.gpus)))]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stopping.set()
            with lock:
                for process in children:
                    process.terminate()
            for future in futures:
                future.cancel()
            raise


def execute_workflow(args, logs, environment):
    if args.task in ('all', 'train'):
        for experts in args.expert_counts:
            print(f'Starting complete stage: {experts} experts', flush=True)
            execute_training(args, experts, logs, environment)
            if args.task == 'all':
                for task in ('trajectories', 'bifurcations'):
                    subprocess.run(evaluation_command(args, task, experts), env=environment, check=True)
            print(f'Finished stage: {experts} experts', flush=True)
        if args.task == 'all' and len(args.expert_counts) > 1:
            # Both stages are complete. Build direct 10-vs-40 diagrams using
            # their validated caches, without repeating simulation or rollouts.
            command = evaluation_command(args, 'bifurcations')
            command = [value for value in command if value != '--force-recompute']
            command.extend(['--merge-expert-caches', str(args.output_root / 'bifurcations')])
            subprocess.run(command, env=environment, check=True)
    else:
        subprocess.run(evaluation_command(args, args.task), env=environment, check=True)


def main(argv=None):
    args = parse_args(argv)
    for gpu in args.gpus:
        configure(args.device, gpu, args.threads)
    data = Trajectories(args.source, args.split_index, args.sequence_length)
    if args.task in ('all', 'trajectories'):
        data.test(args.trajectory_steps)
    if args.task in ('all', 'bifurcations'):
        from evaluate import load_backend
        load_backend(args.analysis_dir).regime_by_name('halvorsen')
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / 'logs'
    logs.mkdir(exist_ok=True)
    write_json(args.output_root / 'experiment_plan.json', {
        'settings': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'data_identity': data.identity, 'splits': SPLITS,
        'stages': [{'experts': experts,
                    'training_commands': [training_command(args, split, experts, args.gpus[i % len(args.gpus)])
                                          for i, split in enumerate(args.splits)],
                    'evaluation_commands': [evaluation_command(args, task, experts)
                                            for task in ('trajectories', 'bifurcations')]}
                   for experts in args.expert_counts],
    })
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                       MKL_NUM_THREADS=str(args.threads))
    execute_workflow(args, logs, environment)
    print(f'Finished {args.task}: {args.output_root}', flush=True)


if __name__ == '__main__':
    main()
