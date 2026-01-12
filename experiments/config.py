"""
Experiment Configurations
=========================

Pre-defined configurations for reproducible experiments.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Base experiment configuration."""

    name: str
    seed: int = 42
    device: str = "auto"
    results_dir: str = "results"


@dataclass
class BSBDimensionConfig(ExperimentConfig):
    """BSB equation across dimensions D = 10, 50, 100, 200."""

    name: str = "bsb_dimension_scaling"
    dimensions: List[int] = field(default_factory=lambda: [10, 50, 100, 200])

    # Training - matching reference implementation
    n_iterations: int = 10000
    batch_size: int = 1
    learning_rate: float = 1e-3

    # Equation
    sigma_min: float = 0.1
    sigma_max: float = 0.3
    terminal_time: float = 1.0

    # Network
    architecture: str = "naisnet"  # 'naisnet' or 'feedforward'
    hidden_dim: int = 256
    num_layers: int = 4


@dataclass
class ArchitectureConfig(ExperimentConfig):
    """Compare NAIS-Net vs FeedForward."""

    name: str = "architecture_comparison"
    architectures: List[str] = field(default_factory=lambda: ["naisnet", "feedforward"])

    dimension: int = 100
    n_iterations: int = 10000
    n_trials: int = 3  # Repeat for statistics


@dataclass
class BlackScholesConfig(ExperimentConfig):
    """Black-Scholes basket option pricing."""

    name: str = "black_scholes_basket"
    dimensions: List[int] = field(default_factory=lambda: [10, 50, 100])

    # Option parameters
    strike: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    terminal_time: float = 1.0
    payoff: str = "basket_call"

    # Training
    n_iterations: int = 10000
    batch_size: int = 1


@dataclass
class GlobalSolverConfig(ExperimentConfig):
    """Global solver with distributed initial conditions."""

    name: str = "global_solver"
    dimension: int = 50
    n_iterations: int = 10000

    # Sampling
    sampling: str = "lognormal"  # 'uniform', 'lognormal', 'gaussian'
    X0_center: float = 100.0
    X0_spread: float = 0.2

    # Evaluation grid
    eval_points: int = 25
    eval_range: tuple = (70.0, 130.0)


# Default configs
BSB_CONFIG = BSBDimensionConfig()
ARCH_CONFIG = ArchitectureConfig()
BS_CONFIG = BlackScholesConfig()
GLOBAL_CONFIG = GlobalSolverConfig()
