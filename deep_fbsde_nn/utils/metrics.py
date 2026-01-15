"""
Metrics and Benchmarks
======================

Error metrics and Monte Carlo benchmarking utilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union


def relative_error(predicted: float, exact: float) -> float:
    """
    Compute relative error as percentage.

    Args:
        predicted: Predicted value
        exact: Exact/reference value

    Returns:
        Relative error in percentage
    """
    if abs(exact) < 1e-10:
        return abs(predicted - exact) * 100
    return abs(predicted - exact) / abs(exact) * 100


def mean_squared_error(predicted: torch.Tensor, exact: torch.Tensor) -> float:
    """Compute MSE."""
    return torch.mean((predicted - exact) ** 2).item()


def max_absolute_error(predicted: torch.Tensor, exact: torch.Tensor) -> float:
    """Compute maximum absolute error."""
    return torch.max(torch.abs(predicted - exact)).item()


def monte_carlo_price(
    S0: Union[float, np.ndarray],
    K: float,
    T: float,
    r: float,
    sigma: float,
    D: int,
    correlation: float = 0.0,
    n_paths: int = 500000,
    payoff_type: str = "basket_call",
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Monte Carlo price for basket options.

    Args:
        S0: Initial spot price(s)
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        D: Number of assets
        correlation: Pairwise correlation
        n_paths: Number of Monte Carlo paths
        payoff_type: 'basket_call', 'basket_put', 'best_of', 'worst_of'
        device: Compute device

    Returns:
        (price, standard_error)
    """
    dev = torch.device(device)

    # Initial prices
    if np.isscalar(S0):
        S0_tensor = torch.full((D,), float(S0), device=dev)
    else:
        S0_tensor = torch.tensor(S0, device=dev, dtype=torch.float32)

    # Correlation structure
    if correlation > 0:
        corr_matrix = np.full((D, D), correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        L = torch.from_numpy(np.linalg.cholesky(corr_matrix)).float().to(dev)
    else:
        L = torch.eye(D, device=dev)

    # Simulate terminal values
    Z = torch.randn(n_paths, D, device=dev)
    Z_corr = torch.matmul(Z, L.t())

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z_corr

    S_T = S0_tensor * torch.exp(drift + diffusion)

    # Compute payoffs
    if payoff_type == "basket_call":
        basket = S_T.mean(dim=1)
        payoffs = F.relu(basket - K)
    elif payoff_type == "basket_put":
        basket = S_T.mean(dim=1)
        payoffs = F.relu(K - basket)
    elif payoff_type == "best_of":
        best = S_T.max(dim=1)[0]
        payoffs = F.relu(best - K)
    elif payoff_type == "worst_of":
        worst = S_T.min(dim=1)[0]
        payoffs = F.relu(worst - K)
    elif payoff_type == "geometric_call":
        log_basket = torch.log(S_T).mean(dim=1)
        basket = torch.exp(log_basket)
        payoffs = F.relu(basket - K)
    else:
        raise ValueError(f"Unknown payoff type: {payoff_type}")

    # Discount and compute statistics
    discount = np.exp(-r * T)
    price = discount * payoffs.mean().item()
    std_error = discount * payoffs.std().item() / np.sqrt(n_paths)

    return price, std_error


def monte_carlo_bsb_solution(
    x: torch.Tensor, t: float, T: float, sigma_max: float, n_paths: int = 100000
) -> Tuple[float, float]:
    """
    Monte Carlo estimate for BSB equation with g(x) = ||x||².

    The exact solution is u(t,x) = ||x||² * exp(σ_max² * (T-t))
    This function verifies via simulation.

    Args:
        x: Initial state (1, D)
        t: Current time
        T: Terminal time
        sigma_max: Maximum volatility
        n_paths: Number of paths

    Returns:
        (estimated_value, standard_error)
    """
    device = x.device
    D = x.shape[1]
    tau = T - t

    # Simulate X_T from X_t using GBM with zero drift, σ_max diffusion
    sqrt_tau = np.sqrt(tau)
    Z = torch.randn(n_paths, D, device=device)

    # X_T = X_t * exp(-0.5 * σ² * τ + σ * √τ * Z)
    X_T = x * torch.exp(-0.5 * sigma_max**2 * tau + sigma_max * sqrt_tau * Z)

    # Terminal values: g(X_T) = ||X_T||²
    g_values = (X_T**2).sum(dim=1)

    # E[g(X_T) | X_t = x]
    mean_value = g_values.mean().item()
    std_error = g_values.std().item() / np.sqrt(n_paths)

    return mean_value, std_error


class ErrorTracker:
    """Track errors during training for convergence analysis."""

    def __init__(self, exact_value: Optional[float] = None):
        self.exact_value = exact_value
        self.losses = []
        self.predictions = []
        self.iterations = []

    def update(self, iteration: int, loss: float, prediction: float):
        """Record a training step."""
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.predictions.append(prediction)

    def get_relative_errors(self) -> np.ndarray:
        """Get relative errors over training."""
        if self.exact_value is None:
            raise ValueError("Exact value not set")
        predictions = np.array(self.predictions)
        return np.abs(predictions - self.exact_value) / self.exact_value * 100

    def final_error(self) -> float:
        """Get final relative error."""
        if self.exact_value is None:
            return np.nan
        return relative_error(self.predictions[-1], self.exact_value)

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "iterations": self.iterations,
            "losses": self.losses,
            "predictions": self.predictions,
            "exact_value": self.exact_value,
            "final_error": self.final_error() if self.exact_value else None,
        }
