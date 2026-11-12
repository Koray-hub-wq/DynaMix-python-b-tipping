"""Train fixed random mixtures on fresh, balanced 200-step trajectory windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import (DEFAULT_SOURCE, SPLITS, FixedMixture, Trajectories,
                    configure, sha256, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--experts', type=int, choices=(10, 40), required=True)
    parser.add_argument('--latent-dim', type=int, default=30)
    parser.add_argument('--pwl-units', type=int, choices=(2,), default=2)
    parser.add_argument('--phi-dim', type=int, choices=(0, 1), default=1)
    parser.add_argument('--weights-seed', type=int, default=8)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--split-index', type=int, default=80000)
    parser.add_argument('--sequence-length', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batches-per-epoch', type=int, default=20)
    parser.add_argument('--teacher-forcing-interval', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--start-lr', type=float, default=0.004)
    parser.add_argument('--end-lr', type=float, default=0.00001)
    parser.add_argument('--noise-level', type=float, default=0.02)
    parser.add_argument('--ssi', type=int, default=25)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.batches_per_epoch, args.ssi,
           args.teacher_forcing_interval) < 1 or args.latent_dim < 3:
        parser.error('Counts must be positive; latent-dim must be at least 3')
    if not 0 <= args.alpha <= 1 or not 0 <= args.noise_level < float('inf'):
        parser.error('Need alpha in [0,1] and finite nonnegative noise')
    if not 0 < args.end_lr <= args.start_lr < float('inf'):
        parser.error('Need 0 < end-lr <= start-lr, both finite')
    return args


def atomic_save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def training_config(args, data, model):
    config = {key: value for key, value in vars(args).items()
              if key not in ('source', 'output', 'device', 'gpu_id', 'threads', 'resume')}
    config.update(architecture='fixed_random_mixture_alrnn', version=1,
                  source=str(data.source), data_identity=data.identity,
                  train_phi_values=SPLITS[args.split], context=False,
                  weights=model.weights.cpu().tolist(), weight_distribution='Dirichlet(1,...,1)',
                  source_sha256={str(path.relative_to(Path(__file__).resolve().parents[2])): sha256(path)
                                 for path in (Path(__file__), Path(__file__).with_name('common.py'),
                                              Path(__file__).resolve().parents[2] / 'src/dynamix/model/fixed_mixture.py',
                                              Path(__file__).resolve().parents[2] / 'src/dynamix/model/dynamix.py')})
    return config


def train(args):
    device = configure(args.device, args.gpu_id, args.threads)
    np.random.seed(args.seed)  # original DynaMix expert initialization uses NumPy
    torch.manual_seed(args.seed)
    data = Trajectories(args.source, args.split_index, args.sequence_length)
    model = FixedMixture(M=args.latent_dim, Experts=args.experts, P=args.pwl_units,
                         phi_dim=args.phi_dim, weights_seed=args.weights_seed)
    config = training_config(args, data, model)
    directory = args.output.resolve()
    config_path = directory / 'config.json'
    checkpoints = directory / 'checkpoints'
    final = checkpoints / 'final_model.pt'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing != config:
            raise ValueError(f'Existing run has different data/code/settings: {directory}; use a new output')
        if final.is_file():
            print(f'Already complete (matching configuration): {directory}', flush=True)
            return
        if not args.resume:
            raise FileExistsError(f'Incomplete run: {directory}; use --resume or a new output')
    elif directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Output is nonempty without a config: {directory}')
    checkpoints.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    model.to(device).train()
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(args.end_lr / args.start_lr) ** (1 / args.epochs))
    # Separate sampling and augmentation generators give both expert counts the
    # same windows and noise, irrespective of the number of model parameters.
    rng = np.random.default_rng(args.seed)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    history, start_epoch = [], 0
    last = checkpoints / 'last_training_state.pt'
    if args.resume and last.is_file():
        state = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rng.bit_generator.state = state['sampling_rng']
        noise_rng.set_state(state['noise_rng'].cpu())
        history, start_epoch = state['history'], state['epoch']
        if not torch.equal(model.weights.cpu(), torch.tensor(config['weights'])):
            raise ValueError('Resume checkpoint weights disagree with original frozen weights')
        print(f'Resuming at epoch {start_epoch + 1}', flush=True)
    original_weights = model.weights.detach().clone()
    print(f'{args.split}: phi={SPLITS[args.split]}, experts={args.experts}, M={model.M}, P={model.P}, phi_dim={model.phi_dim}', flush=True)
    print(f'Frozen weights: {config["weights"]}', flush=True)
    print(f'Train [0,{args.split_index}); held out [{args.split_index},{data.x.shape[1]})', flush=True)
    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        losses = []
        for _ in range(args.batches_per_epoch):
            x, y, phi, _, _ = data.sample_batch(args.split, args.batch_size, rng)
            x, y = torch.tensor(x, device=device), torch.tensor(y, device=device)
            phi = torch.tensor(phi, device=device) if args.phi_dim else None
            if args.noise_level:
                noise = torch.randn(x.shape, generator=noise_rng, device=device) * args.noise_level
                noise[0] = 0  # same clean x0 convention as the context ablation
                x = x + noise
            optimizer.zero_grad(set_to_none=True)
            prediction = model.training_sequence(x, phi, interval=args.teacher_forcing_interval, alpha=args.alpha)
            loss = torch.nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss in epoch {epoch + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if not torch.equal(model.weights, original_weights):
            raise AssertionError('Frozen weights changed during training')
        row = {'epoch': epoch + 1, 'mse': float(np.mean(losses)),
               'lr': scheduler.get_last_lr()[0], 'seconds': time.perf_counter() - started}
        history.append(row)
        print(f'Epoch {epoch + 1}/{args.epochs}: MSE={row["mse"]:.7g}, lr={row["lr"]:.5g}, {row["seconds"]:.2f}s', flush=True)
        if (epoch + 1) % args.ssi == 0 or epoch + 1 == args.epochs:
            atomic_save(checkpoints / f'epoch_{epoch + 1:04d}.pt', model.state_dict())
            atomic_save(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,
                              'history': history, 'sampling_rng': rng.bit_generator.state,
                              'noise_rng': noise_rng.get_state()})
            write_json(directory / 'training_history.json', history)
    atomic_save(final, model.state_dict())
    print(f'Completed: {final}', flush=True)


if __name__ == '__main__':
    train(parse_args())
