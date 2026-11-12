"""Run all six tiny CPU trainings and all plots on real data (not a quality test)."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--analysis-dir', required=True, type=Path)
    args = parser.parse_args()
    subprocess.run([
        sys.executable, str(Path(__file__).with_name('run.py')), '--source', str(args.source.resolve()),
        '--output-root', str(args.output.resolve()), '--analysis-dir', str(args.analysis_dir.resolve()),
        '--device', 'cpu', '--gpus', '0', '--threads', '1', '--epochs', '1',
        '--batches-per-epoch', '1', '--batch-size', '2', '--ssi', '1',
        '--trajectory-steps', '40', '--bifurcation-steps', '1500', '--num-parameters', '3',
        '--burn-in-steps', '100', '--discard-steps', '100',
    ], check=True)


if __name__ == '__main__':
    main()
