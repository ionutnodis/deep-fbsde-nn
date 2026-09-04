"""
Black-Scholes-Barenblatt Equation
=================================

Fully nonlinear PDE from option pricing under uncertain volatility.

PDE:
    ∂u/∂t + (1/2) sup_{σ∈[σ_min,σ_max]} {σ² x² ∂²u/∂x²} = 0
    u(T, x) = g(x)

For convex payoff g(x) = ||x||², σ = σ_max and we have analytical solution:
    u(t, x) = ||x||² · exp(σ_max² · (T - t))

This is the canonical test case from:
- Han, Jentzen, E (2018): "Solving high-dimensional PDEs using deep learning"
- Beck, Jentzen, E (2019): "Machine learning for fully nonlinear PDEs"
"""

from typing import Union

import numpy as np
import torch

from .base import BaseEquation, EquationConfig


class BlackScholesBarenblattEquation(BaseEquation):
    """
    BSB equation with g(x) = ||x||².

    Has analytical solution - ideal for validation.

    Args:
        dimension: D
        sigma_min: Lower vol bound (not used for convex payoff)
        sigma_max: Upper vol bound
        terminal_time: T
    """

    def __init__(
        self,
        dimension: int,
        sigma_min: float = 0.1,
        sigma_max: float = 0.3,
        terminal_time: float = 1.0,
        device: torch.device = None,
    ):
        config = EquationConfig(
            name='Black-Scholes-Barenblatt',
            dimension=dimension,
            terminal_time=terminal_time,
        )
        super().__init__(config, device)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._sigma_max_sq = sigma_max ** 2

    def drift(self, t: torch.Tensor, X: torch.Tensor,
              Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """μ = 0 (pure diffusion)."""
        return torch.zeros_like(X)

    def diffusion(self, t: torch.Tensor, X: torch.Tensor,
                  Y: torch.Tensor) -> torch.Tensor:
        """σ = σ_max · X (convex payoff → max volatility)."""
        return self.sigma_max * X

    def driver(self, t: torch.Tensor, X: torch.Tensor,
               Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """f = 0 (no discounting)."""
        return torch.zeros_like(Y)

    def terminal(self, X: torch.Tensor) -> torch.Tensor:
        """g(x) = ||x||² = Σᵢ xᵢ²"""
        return torch.sum(X ** 2, dim=1, keepdim=True)

    def terminal_gradient(self, X: torch.Tensor) -> torch.Tensor:
        """∇g(x) = 2x"""
        return 2.0 * X

    def exact_solution(self, t: Union[float, torch.Tensor],
                       X: torch.Tensor) -> torch.Tensor:
        """
        u(t, x) = ||x||² · exp(σ_max² · (T - t))
        """
        if isinstance(t, torch.Tensor):
            t = t.item() if t.numel() == 1 else t.flatten()[0].item()

        tau = self.T - t
        norm_sq = torch.sum(X ** 2, dim=1, keepdim=True)
        return norm_sq * np.exp(self._sigma_max_sq * tau)

    def exact_gradient(self, t: Union[float, torch.Tensor],
                       X: torch.Tensor) -> torch.Tensor:
        """∇u(t, x) = 2x · exp(σ_max² · (T - t))"""
        if isinstance(t, torch.Tensor):
            t = t.item() if t.numel() == 1 else t.flatten()[0].item()

        tau = self.T - t
        return 2.0 * X * np.exp(self._sigma_max_sq * tau)

    def sample_initial_condition(self, batch_size: int = 1) -> torch.Tensor:
        """x = (1/√D, ..., 1/√D) so ||x||² = 1."""
        return torch.ones(batch_size, self.D, device=self.device) / np.sqrt(self.D)


class AllenCahnEquation(BaseEquation):
    """
    Allen-Cahn equation — the canonical deep-BSDE benchmark specification.

    PDE (Han, Jentzen, E 2018, PNAS 115(34)):
        ∂u/∂t + Δu + u - u³ = 0
        u(T, x) = 1 / (2 + 0.4·||x||²)
    with dX = √2 dW (so the generator is the full Laplacian Δ) and T = 0.3.

    Reference value (branching-diffusion benchmark, reused across the
    deep-BSDE literature): u(0, 0) ≈ 0.052802 at d = 100.

    Note: v0.1 shipped a nonstandard variant (σ = I, g = 1/√(1+||x||²)) with
    no reference value anywhere; v0.2 aligns with the literature-standard
    test case so the equation is validatable.
    """

    KNOWN_VALUE_D100 = 0.052802  # u(0, 0), branching-diffusion benchmark

    def __init__(
        self,
        dimension: int,
        terminal_time: float = 0.3,
        device: torch.device = None,
    ):
        config = EquationConfig(
            name='Allen-Cahn',
            dimension=dimension,
            terminal_time=terminal_time,
        )
        super().__init__(config, device)
        self.sqrt_2 = np.sqrt(2.0)

    def drift(self, t: torch.Tensor, X: torch.Tensor,
              Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(X)

    def diffusion(self, t: torch.Tensor, X: torch.Tensor,
                  Y: torch.Tensor) -> torch.Tensor:
        """σ = √2 · I (canonical: generator = Δ)."""
        return torch.full_like(X, self.sqrt_2)

    def driver(self, t: torch.Tensor, X: torch.Tensor,
               Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """f = u - u³: the PDE is ∂u/∂t + Δu + (u - u³) = 0, and driver()
        returns f from that form (dY = -f dt + Zᵀσ dW; σ=√2 supplies Δ)."""
        return Y - Y ** 3

    def terminal(self, X: torch.Tensor) -> torch.Tensor:
        """g(x) = 1 / (2 + 0.4·||x||²) — the canonical benchmark payoff."""
        norm_sq = torch.sum(X ** 2, dim=1, keepdim=True)
        return 1.0 / (2.0 + 0.4 * norm_sq)

    def terminal_gradient(self, X: torch.Tensor) -> torch.Tensor:
        """∇g(x) = -0.8·x / (2 + 0.4·||x||²)²"""
        norm_sq = torch.sum(X ** 2, dim=1, keepdim=True)
        return -0.8 * X / (2.0 + 0.4 * norm_sq) ** 2

    def sample_initial_condition(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.D, device=self.device)
