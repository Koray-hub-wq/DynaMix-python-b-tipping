import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GatingNetwork(nn.Module):
    def __init__(self, N, M, Experts, dtype=torch.float32):
        super().__init__()
        self.conv = nn.Conv1d(N, N, kernel_size=2, padding=0, bias=True, dtype=dtype)
        self.softmax_temp1 = nn.Parameter(torch.tensor([0.1], dtype=dtype))
        self.D = nn.Parameter(torch.zeros(N, M, dtype=dtype))
        self.D.data[:, :N] = torch.eye(N, dtype=dtype)
        self.mlp_layer1 = nn.Linear(M + N, Experts, dtype=dtype)
        self.mlp_layer1.bias = nn.Parameter(torch.zeros(Experts, dtype=dtype), requires_grad=False)
        self.mlp_layer2 = nn.Linear(Experts, Experts, dtype=dtype)
        self.mlp_layer2.bias = nn.Parameter(torch.zeros(Experts, dtype=dtype), requires_grad=False)
        self.softmax_temp2 = nn.Parameter(torch.tensor([0.1], dtype=dtype))
        self.sigma = nn.Parameter(torch.ones(N, dtype=dtype) * 0.05, requires_grad=True)

    def forward(self, context, z, precomputed_cnn=None):
        # context: (seq_length, batch_size, N)
        # z: (M, batch_size)
        # precomputed_cnn: Optional precomputed CNN features for inference (seq_length-1, batch_size, N)
        
        seq_length, batch_size, N = context.shape
        M = z.shape[0]
        
        # Compute attention weights
        z_obs = self.D @ z.detach()
        z_current = z_obs + self.sigma.unsqueeze(1) * torch.randn(N, batch_size, dtype=z.dtype, device=z.device)
        
        z_current_t = z_current.transpose(0, 1)
        context_frames = context[:-1]

        distances = torch.sum(torch.abs(context_frames - z_current_t.unsqueeze(0)), dim=2)
        attention_weights = F.softmax(-distances / torch.abs(self.softmax_temp1[0]), dim=0)
        
        # Process context with convolution
        # Use precomputed CNN features if provided, otherwise compute them
        if precomputed_cnn is not None:
            encoded = precomputed_cnn
        else:      
            context_for_conv = context.permute(1, 2, 0)
            encoded = self.conv(context_for_conv)
            encoded = encoded.permute(2, 0, 1)
        
        # Build weighted embedding
        weighted_encoded = encoded * attention_weights.unsqueeze(2)
        embedding = torch.sum(weighted_encoded, dim=0)
        embedding = embedding.transpose(0, 1)
        
        # Predict expert weights
        combined = torch.cat([embedding, z], dim=0)
        combined_t = combined.transpose(0, 1)
        mlp_output = self.mlp_layer2(F.relu(self.mlp_layer1(combined_t)))
        w_exp = F.softmax(-mlp_output.transpose(0, 1) / torch.abs(self.softmax_temp2[0]), dim=0)
        return w_exp
    
    def gaussian_init(self, M, N, dtype=torch.float32):
        return torch.randn(M, N, dtype=dtype) * 0.01

class ExpertNetwork(nn.Module):
    """Base class for different expert architectures."""
    def __init__(self, M, P=0, phi_dim=0, probabilistic=False, dtype=torch.float32):
        super().__init__()
        self.M = M
        self.P = P
        self.phi_dim = phi_dim
        self.probabilistic = probabilistic
        self.dtype = dtype
        if phi_dim > 0:
            self.C = nn.Parameter(torch.zeros(M, phi_dim, dtype=dtype))
        
        # Parameter for probabilistic experts
        if probabilistic:
            self.sigma = nn.Parameter(torch.ones(1, dtype=dtype) * 0.05, requires_grad=True)
    
    def forward(self, z, phi_t=None):
        raise NotImplementedError("Subclasses must implement forward method")

    def external_input(self, phi_t, batch_size):
        if self.phi_dim == 0:
            return 0.0
        if phi_t is None:
            raise ValueError("phi_t must be provided when phi_dim > 0")
        if phi_t.dim() != 2:
            raise ValueError(f"Expected phi_t with shape (phi_dim, batch_size), got {phi_t.shape}")
        if phi_t.shape[0] != self.phi_dim or phi_t.shape[1] != batch_size:
            raise ValueError(
                f"Expected phi_t shape ({self.phi_dim}, {batch_size}), got {phi_t.shape}"
            )
        return self.C @ phi_t
    
    def add_noise(self, z):
        """Add stochasticity to the latent state if in probabilistic mode.
        
        Args:
            z: Input tensor
        """
        if self.probabilistic:
            batch_size = z.shape[1]
            noise = torch.randn(self.M, batch_size, dtype=z.dtype, device=z.device)
            return z + self.sigma * noise
        return z
    
    def gaussian_init(self, M, N):
        return torch.randn(M, N, dtype=self.dtype) * 0.01
    
    def normalized_positive_definite(self, M):
        R = np.random.randn(M, M).astype(np.float32)
        K = R.T @ R / M + np.eye(M)
        lambd = np.max(np.abs(np.linalg.eigvals(K)))
        return K / lambd

class AlmostLinearRNN(ExpertNetwork):
    """Almost linear RNN expert architecture."""
    def __init__(self, M, P, phi_dim=0, probabilistic=False, dtype=torch.float32):
        super().__init__(M, P, phi_dim=phi_dim, probabilistic=probabilistic, dtype=dtype)
        self.A, self.W, self.h = self.initialize_A_W_h(M)
        
    def forward(self, z, phi_t=None):
        # z: (M, batch_size)
        # phi_t: optional external input, shape (phi_dim, batch_size)
        # Split z into regular and ReLU parts
        z1 = z[:-self.P, :]
        z2 = F.relu(z[-self.P:, :])
        zcat = torch.cat([z1, z2], dim=0)
        
        output = (
            self.A.unsqueeze(-1) * z
            + self.W @ zcat
            + self.h.unsqueeze(-1)
            + self.external_input(phi_t, z.shape[1])
        )
        
        # Add stochasticity if probabilistic
        if self.probabilistic:
            output = self.add_noise(output)
            
        return output
    
    def initialize_A_W_h(self, M):
        A = torch.nn.Parameter(torch.diag(torch.tensor(self.normalized_positive_definite(M), dtype=self.dtype)))
        W = torch.nn.Parameter(self.gaussian_init(M, M))
        h = torch.nn.Parameter(torch.zeros(M, dtype=self.dtype))
        return A, W, h

class ClippedShallowPLRNN(ExpertNetwork):
    """Clipped shallow PLRNN expert architecture."""
    def __init__(self, M, hidden_dim=50, phi_dim=0, probabilistic=False, dtype=torch.float32):
        super().__init__(M, hidden_dim, phi_dim=phi_dim, probabilistic=probabilistic, dtype=dtype)
        self.A = torch.nn.Parameter(torch.diag(torch.tensor(self.normalized_positive_definite(M), dtype=self.dtype)))
        self.W1 = torch.nn.Parameter(self.gaussian_init(M, hidden_dim))
        self.W2 = torch.nn.Parameter(self.gaussian_init(hidden_dim, M))
        self.h1 = torch.nn.Parameter(torch.zeros(M, dtype=self.dtype))
        self.h2 = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=self.dtype))
        
    def forward(self, z, phi_t=None):
        # z: (M, batch_size)
        # phi_t: optional external input, shape (phi_dim, batch_size)
        W2z = self.W2 @ z
        output = (self.A.unsqueeze(-1) * z + 
                self.W1 @ (F.relu(W2z + self.h2.unsqueeze(-1)) - F.relu(W2z)) + 
                self.h1.unsqueeze(-1) +
                self.external_input(phi_t, z.shape[1]))
        
        # Add stochasticity if probabilistic
        if self.probabilistic:
            output = self.add_noise(output)
            
        return output

class DynaMix(nn.Module):
    def __init__(self, M, N, Experts, P=2, hidden_dim=50, expert_type="almost_linear_rnn", 
                 phi_dim=0,
                 probabilistic_expert=False, dtype=torch.float32):
        """
        Initialize a DynaMix model.
        
        Args:
            M: Dimension of latent state
            N: Dimension of observation space
            Experts: Number of experts
            P: Number of ReLU dimensions
            hidden_dim: Hidden dimension for clipped shallow PLRNN
            expert_type: Type of expert to use ("almost_linear_rnn" or "clipped_shallow_plrnn")
            phi_dim: Dimension of known external parameter input phi_t
            probabilistic_expert: Whether to use probabilistic experts
            dtype: Data type for model parameters (default: torch.float32)
        """
        super().__init__()
        
        self.expert_type = expert_type
        self.probabilistic_expert = probabilistic_expert
        self.experts = nn.ModuleList()
        self.dtype = dtype
        
        for _ in range(Experts):
            if expert_type == "almost_linear_rnn":
                self.experts.append(AlmostLinearRNN(M, P, phi_dim=phi_dim, probabilistic=probabilistic_expert, dtype=dtype))
            elif expert_type == "clipped_shallow_plrnn":
                self.experts.append(ClippedShallowPLRNN(M, hidden_dim, phi_dim=phi_dim, probabilistic=probabilistic_expert, dtype=dtype))
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
        
        self.gating_network = GatingNetwork(N, M, Experts, dtype=dtype)
        self.B = nn.Parameter(self.uniform_init((N, M), dtype=dtype))
        self.N = N
        self.Experts = Experts
        self.P = P
        self.hidden_dim = hidden_dim
        self.M = M
        self.phi_dim = phi_dim

    def step(self, z, context, phi_t=None, precomputed_cnn=None):
        # z: (M, batch_size)
        # context: (seq_length, batch_size, N)
        # phi_t: optional external input, shape (phi_dim, batch_size)
        # precomputed_cnn: Optional precomputed CNN features

        # Compute expert weights
        w_exp = self.gating_network(context, z, precomputed_cnn=precomputed_cnn)  # (Experts, batch_size)
        results = []
        
        # Compute expert outputs
        for i in range(self.Experts):
            expert_output = self.experts[i](z, phi_t=phi_t)
            results.append(expert_output * w_exp[i, :].unsqueeze(0))
        
        # Combine expert outputs
        return torch.sum(torch.stack(results, dim=0), dim=0)

    def forward(self, z, context, phi_t=None, precomputed_cnn=None):
        """
        Forward pass through the DynaMix model.
        
        Args:
            z: Latent state of shape (M, batch_size)
            context: Context data of shape (seq_length, batch_size, N)
            phi_t: Optional known external input of shape (phi_dim, batch_size)
            precomputed_cnn: Optional precomputed CNN features to avoid redundant computation for inference
            
        Returns:
            Updated latent state
        """
        return self.step(z, context, phi_t=phi_t, precomputed_cnn=precomputed_cnn)
        
    def precompute_cnn(self, context):
        """
        Precompute CNN features for more efficient inference.
        
        Args:
            context: Context data of shape (seq_length, batch_size, N)
            
        Returns:
            Precomputed CNN features
        """
        # Process context with convolution
        context_for_conv = context.permute(1, 2, 0)
        encoded = self.gating_network.conv(context_for_conv)
        
        return encoded.permute(2, 0, 1)
    
    def uniform_init(self, shape, dtype=torch.float32):
        din = shape[-1]
        r = 1 / np.sqrt(din)
        return (torch.rand(shape, dtype=dtype) * 2 - 1) * r
    
    def gaussian_init(self, M, N):
        return torch.randn(M, N, dtype=self.dtype) * 0.01

def print_model_parameters(model):
    """Print simplified breakdown of model parameters by component."""
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "-"*60)
    print("Model Parameter Summary:")
    print(f"  Architecture: DynaMix with {model.expert_type} experts")
    if model.expert_type == "almost_linear_rnn":
        print(f"  Dimensions: M={model.M}, N={model.N}, Experts={model.Experts}, P={model.P}, phi_dim={model.phi_dim}")
    else:
        print(f"  Dimensions: M={model.M}, N={model.N}, Experts={model.Experts}, Hidden dim={model.hidden_dim}, phi_dim={model.phi_dim}")
    print(f"  Probabilistic experts: {model.probabilistic_expert}")
    
    # Count parameters
    gating_params = sum(p.numel() for p in model.gating_network.parameters())
    expert_params = sum(p.numel() for expert in model.experts for p in expert.parameters())
    b_params = model.B.numel()
    
    # Print parameter counts
    print(f"\nParameter counts:")
    print(f"  Gating Network: {gating_params:,} ({gating_params/total_params:.1%})")
    print(f"  Experts: {expert_params:,} ({expert_params/total_params:.1%})")
    print(f"  Observation matrix: {b_params:,} ({b_params/total_params:.1%})")
    print(f"  Total: {total_params:,} parameters")
    print("-"*60)
