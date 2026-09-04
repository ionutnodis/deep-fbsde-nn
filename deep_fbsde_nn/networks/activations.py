"""
Activation Functions
====================

Custom activation functions for Deep BSDE networks.

The Sine activation is particularly effective for PDE problems
due to its smoothness and periodic nature, which helps capture
the smooth structure of PDE solutions.

References:
- Sitzmann et al. (2020): "Implicit Neural Representations with
  Periodic Activation Functions" (SIREN)
"""


import torch
import torch.nn as nn


class Sine(nn.Module):
    """
    Sinusoidal activation function.

    f(x) = sin(ω * x)

    Particularly effective for:
    - PDE solutions (smooth, bounded)
    - Implicit neural representations
    - Problems requiring smooth derivatives

    Args:
        omega: Frequency scaling factor (default 1.0)
               Higher values increase frequency of oscillations
    """

    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)

    def __repr__(self):
        return f"Sine(omega={self.omega})"


class Swish(nn.Module):
    """
    Swish activation: f(x) = x * sigmoid(x)

    Smooth alternative to ReLU with better gradient flow.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Softplus(nn.Module):
    """
    Softplus activation: f(x) = (1/β) * log(1 + exp(β * x))

    Smooth approximation to ReLU.

    Args:
        beta: Sharpness parameter (default 1.0)
              Higher values make it closer to ReLU
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x, beta=self.beta)


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Activation name ('sine', 'relu', 'tanh', 'gelu', 'swish', 'softplus')
        **kwargs: Additional arguments passed to activation constructor

    Returns:
        nn.Module activation function

    Examples:
        >>> act = get_activation('sine', omega=30.0)
        >>> act = get_activation('relu')
        >>> act = get_activation('tanh')
    """
    name = name.lower()

    activations = {
        "sine": lambda: Sine(**kwargs),
        "sin": lambda: Sine(**kwargs),
        "relu": lambda: nn.ReLU(),
        "tanh": lambda: nn.Tanh(),
        "gelu": lambda: nn.GELU(),
        "swish": lambda: Swish(),
        "silu": lambda: nn.SiLU(),  # SiLU is same as Swish
        "softplus": lambda: Softplus(**kwargs),
        "sigmoid": lambda: nn.Sigmoid(),
        "elu": lambda: nn.ELU(),
        "leaky_relu": lambda: nn.LeakyReLU(**kwargs),
    }

    if name not in activations:
        available = ", ".join(activations.keys())
        raise ValueError(f"Unknown activation: {name}. Available: {available}")

    return activations[name]()


# Weight initialization for different activations
def init_weights_sine(module: nn.Module, omega: float = 1.0, is_first: bool = False):
    """
    Initialize weights for Sine activation (SIREN-style).

    For first layer: U(-1/input_dim, 1/input_dim)
    For other layers: U(-sqrt(6/input_dim)/omega, sqrt(6/input_dim)/omega)

    Args:
        module: Linear layer to initialize
        omega: Frequency parameter
        is_first: Whether this is the first layer
    """
    if isinstance(module, nn.Linear):
        input_dim = module.weight.shape[1]

        if is_first:
            bound = 1.0 / input_dim
        else:
            bound = (6.0 / input_dim) ** 0.5 / omega

        nn.init.uniform_(module.weight, -bound, bound)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_xavier(module: nn.Module):
    """
    Xavier/Glorot initialization - good default for tanh/sigmoid.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_kaiming(module: nn.Module):
    """
    Kaiming/He initialization - good for ReLU variants.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
