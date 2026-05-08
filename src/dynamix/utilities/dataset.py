import torch
import numpy as np
from random import randint

class Dataset:
    """
    Dataset class for handling time series data for DynaMix model training.
    
    This class manages training, context, and test data for DynaMix models.
    It handles data batching and provides methods to sample random batches
    for training.
    """

    def __init__(
        self,
        data,
        context,
        test,
        phi=None,
        context_phi=None,
        test_phi=None,
        batch_size=16,
        noise_level=0.05,
        device=None,
        dtype=torch.float32,
    ):
        """
        Initialize the dataset with training, context, and test data.
        
        Args:
            data: Training data with shape (seq_length, num_sequences, N)
                where N is the observation space dimension. Can be numpy array or torch tensor.
            context: Context data with shape (context_length, num_sequences, N)
                used for model initialization during training and prediction.
                Can be numpy array or torch tensor.
            test: Test data with shape (seq_length, num_systems, N)
                used for model evaluation. Can be numpy array or torch tensor.
            phi: Optional external parameter input with shape
                (seq_length, num_sequences, phi_dim).
            context_phi: Optional context-aligned parameter input with shape
                (context_length, num_sequences, phi_dim).
            test_phi: Optional test parameter input with shape
                (test_length, num_systems, phi_dim).
            batch_size: Number of sequences to include in each batch
            noise_level: Standard deviation of Gaussian noise added
            device: Device to store tensors on (e.g., 'cpu', 'cuda')
            dtype: Data type to use for tensors (defaults to torch.float32)
            
        Raises:
            ValueError: If data dimensions are incompatible
        """
        # Set device (default to CPU if not specified)
        self.device = torch.device('cpu') if device is None else device
        self.dtype = dtype
        
        # Convert numpy arrays to torch tensors if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context.astype(np.float32))
        if isinstance(test, np.ndarray):
            test = torch.from_numpy(test.astype(np.float32))
        if isinstance(phi, np.ndarray):
            phi = torch.from_numpy(phi.astype(np.float32))
        if isinstance(context_phi, np.ndarray):
            context_phi = torch.from_numpy(context_phi.astype(np.float32))
        if isinstance(test_phi, np.ndarray):
            test_phi = torch.from_numpy(test_phi.astype(np.float32))
        
        # Validate input types
        if not isinstance(data, torch.Tensor) or not isinstance(context, torch.Tensor) or not isinstance(test, torch.Tensor):
            raise TypeError("data, context, and test must be numpy arrays or torch.Tensor objects")
        
        # Validate input shapes
        if len(data.shape) != 3:
            raise ValueError(f"Expected data to have 3 dimensions (seq_length, num_sequences, N), got shape {data.shape}")
        
        if len(context.shape) != 3:
            raise ValueError(f"Expected context to have 3 dimensions (context_length, num_sequences, N), got shape {context.shape}")
        
        if len(test.shape) != 3:
            raise ValueError(f"Expected test to have 3 dimensions (seq_length, num_systems, N), got shape {test.shape}")
        
        # Validate compatibility between tensors
        if data.shape[1] != context.shape[1]:
            raise ValueError(f"Number of sequences in data ({data.shape[1]}) and context ({context.shape[1]}) must match")
            
        if data.shape[2] != context.shape[2] or data.shape[2] != test.shape[2]:
            raise ValueError(f"Feature dimension N must be consistent: data ({data.shape[2]}), context ({context.shape[2]}), test ({test.shape[2]})")

        if phi is not None:
            if not isinstance(phi, torch.Tensor):
                raise TypeError("phi must be a numpy array, torch.Tensor, or None")
            if len(phi.shape) != 3:
                raise ValueError(f"Expected phi to have 3 dimensions (seq_length, num_sequences, phi_dim), got shape {phi.shape}")
            if phi.shape[0] != data.shape[0] or phi.shape[1] != data.shape[1]:
                raise ValueError(f"phi shape {phi.shape} must match data time/sequence dimensions {data.shape[:2]}")

        if context_phi is not None:
            if not isinstance(context_phi, torch.Tensor):
                raise TypeError("context_phi must be a numpy array, torch.Tensor, or None")
            if len(context_phi.shape) != 3:
                raise ValueError(f"Expected context_phi to have 3 dimensions, got shape {context_phi.shape}")
            if context_phi.shape[0] != context.shape[0] or context_phi.shape[1] != context.shape[1]:
                raise ValueError(f"context_phi shape {context_phi.shape} must match context time/sequence dimensions {context.shape[:2]}")

        if test_phi is not None:
            if not isinstance(test_phi, torch.Tensor):
                raise TypeError("test_phi must be a numpy array, torch.Tensor, or None")
            if len(test_phi.shape) != 3:
                raise ValueError(f"Expected test_phi to have 3 dimensions, got shape {test_phi.shape}")
            if test_phi.shape[0] != test.shape[0] or test_phi.shape[1] != test.shape[1]:
                raise ValueError(f"test_phi shape {test_phi.shape} must match test time/sequence dimensions {test.shape[:2]}")

        if phi is not None and context_phi is not None and phi.shape[2] != context_phi.shape[2]:
            raise ValueError("phi and context_phi must have the same phi_dim")
        if phi is not None and test_phi is not None and phi.shape[2] != test_phi.shape[2]:
            raise ValueError("phi and test_phi must have the same phi_dim")
            
        # Store the data on the specified device with the specified dtype
        self.X = data.clone().detach().to(device=self.device, dtype=self.dtype)  # Training data: (seq_length, num_sequences, N)
        self.context = context.clone().detach().to(device=self.device, dtype=self.dtype)  # Context data: (context_length, num_sequences, N)
        self.test = test.clone().detach().to(device=self.device, dtype=self.dtype)  # Test data: (seq_length, num_systems, N)
        self.phi = phi.clone().detach().to(device=self.device, dtype=self.dtype) if phi is not None else None
        self.context_phi = context_phi.clone().detach().to(device=self.device, dtype=self.dtype) if context_phi is not None else None
        self.test_phi = test_phi.clone().detach().to(device=self.device, dtype=self.dtype) if test_phi is not None else None
        
        # Store parameters
        self.batch_size = batch_size
        self.noise_level = noise_level
        
        # Extract dimensions from data for convenience
        self.seq_length = self.X.shape[0]
        self.num_sequences = self.X.shape[1]
        self.context_length = self.context.shape[0]
        self.feature_dim = self.X.shape[2]
        self.phi_dim = 0 if self.phi is None else self.phi.shape[2]
        
    def __getitem__(self, idx):
        """
        Get a single training example by index.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            x: Input sequence (seq_length-1, N) - all but last timestep
            y: Target sequence (seq_length-1, N) - all but first timestep
            context: Context sequence (context_length, N) for this example
            phi: Optional external input sequence (seq_length-1, phi_dim)
        """
        # Validate index
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_sequences} sequences")
        
        # Extract the input sequence (all but last timestep)
        x = self.X[0:self.seq_length-1, idx, :]  # Shape: (seq_length-1, N)
        
        # Extract the target sequence (all but first timestep) 
        y = self.X[1:self.seq_length, idx, :]  # Shape: (seq_length-1, N)
        
        # Extract the context for this sequence
        context = self.context[:, idx, :]  # Shape: (context_length, N)

        if self.phi is None:
            return x, y, context

        phi = self.phi[0:self.seq_length-1, idx, :]  # Shape: (seq_length-1, phi_dim)
        return x, y, context, phi

    def sample_batch(self):
        """
        Sample a batch of sequences randomly from the dataset.
        
        This method randomly selects batch_size sequences and returns them
        as batched tensors ready for model training.
        
        Returns:
            X: Batch of input sequences, shape (seq_length-1, batch_size, N)
            Y: Batch of target sequences, shape (seq_length-1, batch_size, N)
            Context: Batch of context sequences, shape (context_length, batch_size, N)
            Phi: Optional batch of external inputs, shape (seq_length-1, batch_size, phi_dim)
        """
        X = []
        Y = []
        Context = []
        Phi = []
        
        # Sample batch_size random sequences
        for _ in range(self.batch_size):
            # Get a random sequence index
            idx = randint(0, self.num_sequences-1)
            
            # Get the sequence data
            sample = self[idx]
            if self.phi is None:
                x, y, context = sample
            else:
                x, y, context, phi = sample
            
            # Append to lists
            X.append(x)
            Y.append(y)
            Context.append(context)
            if self.phi is not None:
                Phi.append(phi)

        X = torch.stack(X).permute(1, 0, 2)  # Shape: (seq_length-1, batch_size, N)
        Y = torch.stack(Y).permute(1, 0, 2)  # Shape: (seq_length-1, batch_size, N)
        Context = torch.stack(Context).permute(1, 0, 2)  # Shape: (context_length, batch_size, N)
        
        # Add Gaussian noise
        X = X + torch.randn_like(X) * self.noise_level

        if self.phi is None:
            return X, Y, Context

        Phi = torch.stack(Phi).permute(1, 0, 2)  # Shape: (seq_length-1, batch_size, phi_dim)
        return X, Y, Context, Phi
