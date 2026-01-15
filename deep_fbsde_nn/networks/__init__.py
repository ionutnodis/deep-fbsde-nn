"""
Neural Network Architectures
============================

This module provides neural network architectures for Deep BSDE solvers.

Networks:
- NAISNet: Non-Autonomous Input-Output Stable Network (recommended)
- FeedForwardNet: Standard feedforward network (baseline)

Activations:
- Sine: Sinusoidal activation function (best for PDEs)
- get_activation: Factory function for activations

References:
- Ciccone et al. (2018): NAIS-Net paper
- Güler, Laignelet, Parpas (2019): Application to FBSDEs
"""

from .activations import Sine, get_activation, init_weights_xavier
from .naisnet import NAISNet, FeedForwardNet
from .wrapper import BlackScholesWrapper

__all__ = [
    "NAISNet",
    "FeedForwardNet",
    "Sine",
    "get_activation",
    "init_weights_xavier",
    "BlackScholesWrapper",
]
