"""Short CPU integration run on real data; does not measure model quality."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from prepare import prepare, SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--analysis-dir', required=True, type=Path)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    output = args.output.resolve()
    data = output / 'datasets'
    models = output / 'models'
    prepare(args.source, data, test_length=560, windows_per_trajectory=1)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    environment['PYTHONPATH'] = str(directory.parents[1] / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    for split in SPLITS:
        for mode in ('fixed', 'matched'):
            source = data / split / mode
            run = models / split / mode
            log = output / f'{split}_{mode}.log'
            command = [sys.executable, '-m', 'dynamix.training.training_setup',
                       '--data_path', str(source / 'data.npy'), '--context_path', str(source / 'context.npy'),
                       '--phi_path', str(source / 'phi.npy'), '--context_phi_path', str(source / 'context_phi.npy'),
                       '--test_path', str(data / 'evaluation/test.npy'), '--test_phi_path', str(data / 'evaluation/test_phi.npy'),
                       '--test_context_path', str(data / f'evaluation/{mode}_context.npy'),
                       '--test_initial_index', '500', '--no_eval_standardize', '--clean_initial_state',
                       '--save_path', str(run), '--device', 'cpu', '--threads', '1',
                       '--experts', '30', '--latent_dim', '10', '--pwl_units', '2', '--phi_dim', '1',
                       '--epochs', '1', '--batch_size', '2', '--batches_per_epoch', '1', '--ssi', '1',
                       '--seed', '8', '--noise_level', '0.02', '--reg_strength', '0.1']
            print(f'Smoke training: {split}/{mode}; log: {log}', flush=True)
            with log.open('w', encoding='utf-8') as stream:
                subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    shared = ['--data-root', str(data), '--run-root', str(models), '--device', 'cpu', '--threads', '1']
    subprocess.run([sys.executable, str(directory / 'evaluate.py'), 'trajectories', *shared,
                    '--output', str(output / 'trajectories'), '--forecast-steps', '40'], env=environment, check=True)
    subprocess.run([sys.executable, str(directory / 'evaluate.py'), 'bifurcations', *shared,
                    '--analysis-dir', str(args.analysis_dir.resolve()), '--output', str(output / 'bifurcations'),
                    '--forecast-steps', '80', '--discard-steps', '10', '--burn-in-steps', '10',
                    '--num-parameters', '2'], env=environment, check=True)
    print(f'Smoke test completed: {output}', flush=True)


if __name__ == '__main__':
    main()
