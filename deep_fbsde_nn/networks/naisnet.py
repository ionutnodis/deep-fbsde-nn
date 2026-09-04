"""
NAIS-Net: Non-Autonomous Input-Output Stable Network
=====================================================

Implementation based on:
- Ciccone et al. (2018): "NAIS-Net: Stable Deep Networks from Non-Autonomous DEs"
- Güler, Laignelet, Parpas (2019): "Towards robust and stable deep learning for FBSDEs"

NAIS-Net enforces stability via spectral projection of each block's
weight matrix.

Projection mechanism:
    Given weight matrix W, compute:
    1. R^T R = W^T W
    2. If ||R^T R||_F > δ, rescale linearly: R^T R ← δ · R^T R / ||R^T R||_F
    3. A = R^T R + ε·I
    4. Block: h' = activation(-A @ h + b + skip(input)) + h

Since the Frobenius norm bounds the spectral norm and R^T R is symmetric
positive semi-definite, the eigenvalues of A lie in [ε, 1−ε]: A is positive
definite with spectral norm strictly below one. This is the conditioning on
the state matrix used in the stability analysis of Güler, Laignelet &
Parpas (2019).

Parameters:
    - ε (epsilon): Regularization, default 0.01
    - δ (delta): Contraction threshold, δ = 1 - 2ε = 0.98
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import get_activation, init_weights_xavier


class NAISNet(nn.Module):
    """
    NAIS-Net with stability for FBSDE problems.

    All hidden layers have the SAME dimension (required for residual connections).

    Args:
        input_dim: Input dimension (D+1 for time + state)
        hidden_dim: Hidden layer dimension (uniform across all layers)
        output_dim: Output dimension (1 for scalar PDEs)
        num_layers: Total number of hidden layers
        epsilon: Stability parameter (default 0.01)
        activation: Activation function (default 'sine')

    Example:
        >>> # For D=100 dimensional PDE
        >>> net = NAISNet(input_dim=101, hidden_dim=256, output_dim=1, num_layers=4)
        >>> x = torch.randn(64, 101)
        >>> y = net(x)  # (64, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        epsilon: float = 0.01,
        activation: Union[str, nn.Module] = 'sine',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.delta = 1.0 - 2.0 * epsilon

        # Activation
        if isinstance(activation, str):
            self.activation = get_activation(activation)
        else:
            self.activation = activation

        # First layer: input → hidden (standard, no projection)
        self.first_layer = nn.Linear(input_dim, hidden_dim)

        # NAIS blocks: (num_layers - 1) projected layers
        self.nais_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.nais_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.skip_layers.append(nn.Linear(input_dim, hidden_dim))

        # Output layer: hidden → output (standard, no projection)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Identity matrix for projection (buffer, not parameter)
        self.register_buffer('_I', torch.eye(hidden_dim))

        # Initialize
        self.apply(init_weights_xavier)

    def _project(self, W: torch.Tensor) -> torch.Tensor:
        """
        Spectral projection for stability.

        Rescales R^T R = W^T W so that ||R^T R||_F <= δ, then returns
        A = R^T R + ε·I. Because the Frobenius norm bounds the spectral
        norm and R^T R is symmetric PSD, the eigenvalues of A lie in
        [ε, δ+ε] = [ε, 1−ε].
        """
        WtW = W.t() @ W
        norm = torch.norm(WtW)

        if norm > self.delta:
            WtW = WtW * (self.delta / norm)

        return WtW + self._I * self.epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        u = x  # Store input for skip connections

        # First layer
        h = self.activation(self.first_layer(x))

        # NAIS blocks
        for nais, skip in zip(self.nais_layers, self.skip_layers):
            h_prev = h
            A = self._project(nais.weight)
            h = F.linear(h, -A, nais.bias) + skip(u)
            h = self.activation(h) + h_prev

        return self.output_layer(h)

    @property
    def is_stable(self) -> bool:
        """
        Verify the projection invariant on every NAIS block.

        True iff each projected block matrix A = _project(W) has spectral
        norm at most 1 − ε (up to float tolerance). Computed, not assumed.
        """
        bound = 1.0 - self.epsilon + 1e-5
        with torch.no_grad():
            for layer in self.nais_layers:
                A = self._project(layer.weight)
                if torch.linalg.matrix_norm(A, ord=2) > bound:
                    return False
        return True

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"NAISNet(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, layers={self.num_layers}, "
            f"ε={self.epsilon}, params={self.count_parameters():,})"
        )


class FeedForwardNet(nn.Module):
    """
    Standard feedforward network (baseline, no stability guarantees).

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        activation: Union[str, nn.Module] = 'sine',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if isinstance(activation, str):
            self.activation = get_activation(activation)
        else:
            self.activation = activation

        # Build layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(init_weights_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.activation(layer(h))
        return self.output_layer(h)

    @property
    def is_stable(self) -> bool:
        return False

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"FeedForwardNet(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, layers={self.num_layers}, "
            f"params={self.count_parameters():,})"
        )
