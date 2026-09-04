"""
Black-Scholes Equation
======================

Linear PDE for option pricing on multiple assets.

PDE:
    ∂u/∂t + r·x·∇u + (1/2)Tr(σσᵀ D²u) - ru = 0
    u(T, x) = g(x)

FBSDE formulation:
    dX = rX dt + σX dW
    dY = rY dt + Z·σX dW
    Y_T = g(X_T)

Supports basket calls, puts, best-of, worst-of options.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseEquation, EquationConfig


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via torch.erf (keeps core free of scipy)."""
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


class BlackScholesEquation(BaseEquation):
    """
    Black-Scholes for basket options.

    Matches reference implementation:
    - Payoff uses sum(X) not mean(X)
    - Strike defaults to D (ATM when X0=1.0 per asset)
    - X0 = 1.0 per asset
    - r = 0.01, sigma = 0.25 (reference defaults)

    Args:
        dimension: Number of assets D
        r: Risk-free rate (default 0.01)
        sigma: Volatility (default 0.25)
        correlation: Correlation (None, scalar ρ, or matrix)
        strike: Strike price K (default = D for ATM)
        terminal_time: T
        payoff: 'basket_call', 'basket_put', 'geometric_call', 'best_of', 'worst_of'
    """

    def __init__(
        self,
        dimension: int,
        r: float = 0.01,
        sigma: Union[float, np.ndarray] = 0.25,
        correlation: Optional[Union[float, np.ndarray]] = None,
        strike: Optional[float] = None,  # Default to D
        terminal_time: float = 1.0,
        payoff: str = "basket_call",
        device: torch.device = None,
    ):
        config = EquationConfig(
            name=f"Black-Scholes ({payoff})",
            dimension=dimension,
            terminal_time=terminal_time,
        )
        super().__init__(config, device)

        self.r = r
        # Strike = D by default (ATM when each asset starts at 1.0)
        self.strike = float(dimension) if strike is None else strike
        self.payoff_type = payoff
        self.X0_val = 1.0  # Each asset starts at 1.0

        # Volatility
        if np.isscalar(sigma):
            self._sigma_val = sigma
            self._sigma = torch.full((dimension,), sigma, device=self.device)
        else:
            self._sigma_val = sigma[0] if len(sigma) > 0 else 0.25
            self._sigma = torch.tensor(sigma, device=self.device, dtype=torch.float32)

        # Correlation (Cholesky decomposition for correlated BM)
        self._setup_correlation(correlation)

    def _setup_correlation(self, correlation):
        """Setup correlation matrix and Cholesky factor."""
        if correlation is None:
            self.L = torch.eye(self.D, device=self.device)
            self.has_correlation = False
        elif np.isscalar(correlation):
            corr = np.full((self.D, self.D), correlation)
            np.fill_diagonal(corr, 1.0)
            self.L = torch.linalg.cholesky(
                torch.tensor(corr, device=self.device, dtype=torch.float32)
            )
            self.has_correlation = True
        else:
            self.L = torch.linalg.cholesky(
                torch.tensor(correlation, device=self.device, dtype=torch.float32)
            )
            self.has_correlation = True

    @property
    def sigma(self) -> torch.Tensor:
        return self._sigma

    def drift(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """μ = rX (risk-neutral)."""
        return self.r * X

    def diffusion(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """σ = σ · X (diagonal)."""
        return self._sigma_val * X

    def driver(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """f = -rY: the PDE is ∂u/∂t + rx·∇u + (1/2)Tr(σσᵀD²u) - ru = 0,
        so under dY = -f dt + Zᵀσ dW the value process gains +rY dt
        (risk-neutral discounting)."""
        return -self.r * Y

    def terminal(self, X: torch.Tensor) -> torch.Tensor:
        """Payoff g(X) - uses sum(X) like reference."""
        if self.payoff_type == "basket_call":
            # Reference: max(sum(X) - K, 0)
            basket = X.sum(dim=1, keepdim=True)

            # --- FIX 3: Softplus for Gradient Flow ---
            # Use Softplus (smooth ReLU) to ensure non-zero gradients
            # even for OTM paths during training.
            # beta=5.0 gives a sharp approximation that still has a tail.
            return F.softplus(basket - self.strike, beta=5.0)

        elif self.payoff_type == "basket_put":
            basket = X.sum(dim=1, keepdim=True)
            # Apply similar smoothing for Put
            return F.softplus(self.strike - basket, beta=5.0)

        elif self.payoff_type == "geometric_call":
            log_geo = torch.log(X.clamp(min=1e-8)).mean(dim=1, keepdim=True)
            return F.relu(torch.exp(log_geo) * self.D - self.strike)

        elif self.payoff_type == "best_of":
            return F.relu(X.max(dim=1, keepdim=True)[0] * self.D - self.strike)

        elif self.payoff_type == "worst_of":
            return F.relu(X.min(dim=1, keepdim=True)[0] * self.D - self.strike)

        else:
            raise ValueError(f"Unknown payoff: {self.payoff_type}")

    def terminal_gradient(self, X: torch.Tensor) -> torch.Tensor:
        """∇g(X) - analytical for simple payoffs."""
        if self.payoff_type == "basket_call":
            basket = X.sum(dim=1, keepdim=True)
            indicator = (basket > self.strike).float()
            # Gradient of sum is 1 for each component
            return torch.ones_like(X) * indicator

        elif self.payoff_type == "basket_put":
            basket = X.sum(dim=1, keepdim=True)
            indicator = (basket < self.strike).float()
            return -torch.ones_like(X) * indicator

        else:
            return super().terminal_gradient(X)

    def sample_initial_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Start at X0 = 1.0 per asset (like reference)."""
        return torch.full((batch_size, self.D), self.X0_val, device=self.device)

    def monte_carlo_price(self, n_paths: int = 100000) -> tuple:
        """
        Monte Carlo benchmark.

        Returns:
            (price, standard_error)
        """
        S0 = self.X0_val

        # Sample terminal prices
        Z = torch.randn(n_paths, self.D, device=self.device)
        if self.has_correlation:
            Z = Z @ self.L.T

        drift = (self.r - 0.5 * self._sigma_val**2) * self.T
        diff = self._sigma_val * np.sqrt(self.T) * Z
        S_T = S0 * torch.exp(drift + diff)

        # Discounted payoffs
        payoffs = self.terminal(S_T).squeeze()
        discount = np.exp(-self.r * self.T)

        price = discount * payoffs.mean().item()
        stderr = discount * payoffs.std().item() / np.sqrt(n_paths)

        return price, stderr


class VanillaCallEquation(BaseEquation):
    """
    Vanilla Call Option.

    This uses:
    - sum(X) payoff
    - strike = D (each asset starts at 1.0)
    - r = 0.01, sigma = 0.25

    Use this to verify the Deep BSDE implementation against the reference.
    """

    def __init__(
        self,
        dimension: int,
        r: float = 0.01,
        sigma: float = 0.25,
        terminal_time: float = 1.0,
        device: torch.device = None,
    ):
        config = EquationConfig(
            name="Vanilla Call Option",
            dimension=dimension,
            terminal_time=terminal_time,
        )
        super().__init__(config, device)

        self.r = r
        self._sigma_val = sigma
        self._sigma = torch.full((dimension,), sigma, device=self.device)
        self.strike = float(dimension)  # Strike = D (ATM when X0 = 1.0 each)
        self.X0_val = 1.0  # Each asset starts at 1.0

    def drift(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """μ = rX."""
        return self.r * X

    def diffusion(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """σ = σ * X (diagonal)."""
        return self._sigma_val * X

    def driver(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """f = -rY (see BlackScholesEquation.driver for the sign convention)."""
        return -self.r * Y

    def terminal(self, X: torch.Tensor) -> torch.Tensor:
        """g(X) = max(sum(X) - K, 0)."""
        basket_sum = X.sum(dim=1, keepdim=True)
        return F.relu(basket_sum - self.strike)

    def terminal_gradient(self, X: torch.Tensor) -> torch.Tensor:
        """∇g(X) = 1 if sum(X) > K, else 0."""
        basket_sum = X.sum(dim=1, keepdim=True)
        indicator = (basket_sum > self.strike).float()
        return torch.ones_like(X) * indicator

    def sample_initial_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Start at X0 = 1.0 for each asset."""
        return torch.full((batch_size, self.D), self.X0_val, device=self.device)

    def exact_solution(self, t: float, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Black-Scholes closed form — exact only for D=1.

        For D>1 the payoff max(sum(X)−K, 0) is a basket option with no
        closed form; returns None (use monte_carlo_price as a benchmark).

        Args:
            t: Current time
            X: (M, 1) spot prices (must be positive)
        Returns:
            (M, 1) call values, or None when D != 1
        """
        if self.D != 1:
            return None

        tau = self.T - float(t)
        S = X[:, :1]
        K = self.strike
        if tau <= 0:
            return F.relu(S - K)

        sqrt_tau = np.sqrt(tau)
        d1 = (torch.log(S / K) + (self.r + 0.5 * self._sigma_val**2) * tau) / (
            self._sigma_val * sqrt_tau
        )
        d2 = d1 - self._sigma_val * sqrt_tau
        discount = np.exp(-self.r * tau)
        return S * _norm_cdf(d1) - K * discount * _norm_cdf(d2)

    def has_exact_solution(self) -> bool:
        """The closed form applies only to the single-asset case."""
        return self.D == 1

    def monte_carlo_price(self, n_paths: int = 100000) -> tuple:
        """Monte Carlo benchmark for sum-based call."""
        S0 = self.X0_val

        Z = torch.randn(n_paths, self.D, device=self.device)
        drift = (self.r - 0.5 * self._sigma_val**2) * self.T
        diff = self._sigma_val * np.sqrt(self.T) * Z
        S_T = S0 * torch.exp(drift + diff)

        payoffs = self.terminal(S_T).squeeze()
        discount = np.exp(-self.r * self.T)

        price = discount * payoffs.mean().item()
        stderr = discount * payoffs.std().item() / np.sqrt(n_paths)

        return price, stderr
