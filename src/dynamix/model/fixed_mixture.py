"""DynaMix AL-RNN experts with a frozen random convex mixture, without context."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .dynamix import AlmostLinearRNN


class FixedMixture(nn.Module):
    """All experts act on the shared state z; only their parameters are trained.

    z_next = sum_i weights[i] * (A_i*z + W_i*relu_P(z) + h_i + C_i*phi).
    Weights are sampled once from Dirichlet(1,...,1), independently of the
    global initialization RNG, and stored as a checkpoint buffer. There is no
    gating network, context encoder, or context argument.
    """

    def __init__(self, M=30, N=3, Experts=10, P=2, phi_dim=1,
                 weights_seed=8, dtype=torch.float32):
        super().__init__()
        if Experts < 1 or not 1 <= N <= M or not 1 <= P <= M or phi_dim < 0:
            raise ValueError('Require Experts>=1, M>=N>=1, 1<=P<=M and phi_dim>=0')
        self.M, self.N, self.Experts, self.P = M, N, Experts, P
        self.phi_dim = phi_dim
        self.experts = nn.ModuleList([
            AlmostLinearRNN(M, P, phi_dim=phi_dim, dtype=dtype) for _ in range(Experts)
        ])
        radius = M ** -0.5
        self.B = nn.Parameter((2 * torch.rand(N, M, dtype=dtype) - 1) * radius)
        generator = torch.Generator().manual_seed(weights_seed)
        draws = torch.empty(Experts, dtype=dtype).exponential_(generator=generator)
        self.register_buffer('weights', draws / draws.sum())

    def effective_parameters(self):
        """Exact differentiable reduction; compute once per sequence, not per step.

        Shared ReLU coordinates make the fixed mixture another AL-RNN. Keeping
        this reduction in the autograd graph still trains every original expert.
        """
        def mix(name):
            stacked = torch.stack([getattr(expert, name) for expert in self.experts])
            shape = (-1,) + (1,) * (stacked.ndim - 1)
            return (self.weights.reshape(shape) * stacked).sum(dim=0)

        return mix('A'), mix('W'), mix('h'), mix('C') if self.phi_dim else None

    def initial_state(self, initial_x):
        if initial_x.ndim != 2 or initial_x.shape[1] != self.N:
            raise ValueError(f'Expected initial_x (batch, {self.N})')
        hidden = initial_x @ self.B[:, self.N:]
        return torch.cat((initial_x, hidden), dim=1).t()

    def step(self, z, phi_t=None, *, parameters=None):
        if z.ndim != 2 or z.shape[0] != self.M:
            raise ValueError(f'Expected z ({self.M}, batch)')
        A, W, h, C = self.effective_parameters() if parameters is None else parameters
        activated = torch.cat((z[:-self.P], F.relu(z[-self.P:])), dim=0)
        result = A[:, None] * z + W @ activated + h[:, None]
        if self.phi_dim:
            if phi_t is None or phi_t.shape != (self.phi_dim, z.shape[1]):
                raise ValueError(f'Expected phi_t ({self.phi_dim}, {z.shape[1]})')
            result = result + C @ phi_t
        elif phi_t is not None:
            raise ValueError('This model has no phi input')
        return result

    def forward(self, z, phi_t=None):
        return self.step(z, phi_t)

    def training_sequence(self, x, phi, *, interval=10, alpha=1.0):
        """x[t] -> y[t]=x_original[t+1], with sparse teacher forcing."""
        if interval < 1 or not 0 <= alpha <= 1:
            raise ValueError('Invalid teacher forcing interval or strength')
        if x.ndim != 3 or x.shape[2] != self.N:
            raise ValueError(f'Expected x (time, batch, {self.N})')
        if self.phi_dim and (phi is None or phi.shape != (*x.shape[:2], self.phi_dim)):
            raise ValueError('phi must match the time/batch dimensions of x')
        parameters = self.effective_parameters()
        z = self.initial_state(x[0])
        predictions = []
        for t in range(len(x)):
            if t > 0 and t % interval == 0:
                observed = alpha * x[t].t() + (1 - alpha) * z[:self.N]
                z = torch.cat((observed, z[self.N:]), dim=0)
            z = self.step(z, None if phi is None else phi[t].t(), parameters=parameters)
            predictions.append(z[:self.N].t())
        return torch.stack(predictions)

    @torch.no_grad()
    def forecast(self, initial_x, phi_values, horizon):
        """Deterministic free rollout, initialized only from a single observation.

        phi_values: (batch, phi_dim), constant over this forecast. The returned
        first point is one model step after initial_x. No teacher forcing.
        """
        if horizon < 1:
            raise ValueError('horizon must be positive')
        z = self.initial_state(initial_x)
        parameters = self.effective_parameters()
        phi_t = None if phi_values is None else phi_values.t()
        result = torch.empty(horizon, len(initial_x), self.N, device=z.device, dtype=z.dtype)
        for t in range(horizon):
            z = self.step(z, phi_t, parameters=parameters)
            result[t] = z[:self.N].t()
        return result
