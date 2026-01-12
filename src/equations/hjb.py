"""
Hamilton-Jacobi-Bellman (HJB) Equation
======================================

High-dimensional nonlinear PDE arising in stochastic optimal control.
This is the classic LQG (Linear Quadratic Gaussian) control problem benchmark.

PDE:
    ∂u/∂t + Δu - λ||∇u||² = 0
    u(T, x) = ln(0.5 * (1 + ||x||²))

FBSDE formulation:
    dX_t = √2 dW_t
    dY_t = -f(t, X, Y, Z) dt + Z·dW_t

    where f(z) = -λ ||z||² / 2 (derived from Z = √2∇u)

Reference:
    Han, Jentzen, E (2018): "Solving high-dimensional PDEs using deep learning"
"""

import torch
import numpy as np
from typing import Union
from .base import BaseEquation, EquationConfig


class HJBEquation(BaseEquation):
    """
    HJB Equation for LQG Control.

    Args:
        dimension: Problem dimension D (default 100)
        lambda_val: Control cost parameter λ (default 1.0)
        terminal_time: T (default 1.0)
    """

    def __init__(
        self,
        dimension: int = 100,
        lambda_val: float = 1.0,
        terminal_time: float = 1.0,
        device: torch.device = None,
    ):
        config = EquationConfig(
            name="Hamilton-Jacobi-Bellman",
            dimension=dimension,
            terminal_time=terminal_time,
        )
        super().__init__(config, device)

        self.lambda_val = lambda_val
        self.sqrt_2 = np.sqrt(2.0)

        # X0 is typically 0 for this benchmark
        self.X0_val = 0.0

    def drift(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """μ = 0 (Standard Brownian Motion drift)."""
        return torch.zeros_like(X)

    def diffusion(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """σ = √2 * I."""
        return torch.full_like(X, self.sqrt_2)

    def driver(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """
        f = -λ ||∇u||²
        Since Z = σᵀ∇u = √2∇u, we have ∇u = Z/√2.
        Therefore f = -λ ||Z/√2||² = -λ/2 ||Z||².
        """
        norm_z_sq = torch.sum(Z**2, dim=1, keepdim=True)
        return -0.5 * self.lambda_val * norm_z_sq

    def terminal(self, X: torch.Tensor) -> torch.Tensor:
        """g(x) = ln(0.5 * (1 + ||x||²))"""
        norm_x_sq = torch.sum(X**2, dim=1, keepdim=True)
        return torch.log(0.5 * (1.0 + norm_x_sq))

    def terminal_gradient(self, X: torch.Tensor) -> torch.Tensor:
        """
        ∇g(x) = 2x / (1 + ||x||²)
        Used for terminal gradient loss.
        """
        norm_x_sq = torch.sum(X**2, dim=1, keepdim=True)
        return 2.0 * X / (1.0 + norm_x_sq)

    def sample_initial_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Start at X0 = 0."""
        return torch.zeros(batch_size, self.D, device=self.device)

    def exact_solution(
        self, t: float, X: torch.Tensor = None, n_mc: int = 100000
    ) -> torch.Tensor:
        """
        Compute 'exact' solution via Monte Carlo formula.

        u(t, x) = -1/λ * ln( E[ exp(-λ * g(x + √2 * W_{T-t})) ] )

        Args:
            t: Current time
            X: State (if None, uses X0)
            n_mc: Number of MC paths for estimation
        """
        if X is None:
            X = self.sample_initial_condition(1)

        # If calculating for a batch of X, we need to be careful with memory
        # Here we implement for a single X0 or small batch

        tau = self.T - t
        eff_sigma = self.sqrt_2 * np.sqrt(tau)

        # We need independent noise for each sample in X
        # Shape: (n_mc, batch_size, D)
        batch_size = X.shape[0]

        # Monte Carlo estimation
        # We calculate E[...] term

        term_sum = torch.zeros(batch_size, 1, device=self.device)

        # Process in chunks to avoid OOM if n_mc is large
        chunk_size = 10000
        for _ in range(0, n_mc, chunk_size):
            current_n = min(chunk_size, n_mc)

            # Noise: (current_n, batch_size, D)
            Z = torch.randn(current_n, batch_size, self.D, device=self.device)

            # X_T = x + √2 * √tau * Z
            # X shape: (batch_size, D) -> broadcast to (current_n, batch_size, D)
            X_T = X.unsqueeze(0) + eff_sigma * Z

            # g(X_T) shape: (current_n, batch_size, 1)
            # Flatten X_T for terminal computation to match (N, D) expected input
            X_T_flat = X_T.view(-1, self.D)
            g_val_flat = self.terminal(X_T_flat)
            g_val = g_val_flat.view(current_n, batch_size, 1)

            # Accumulate exp(-λ * g)
            term_sum += torch.sum(torch.exp(-self.lambda_val * g_val), dim=0)

        # Average
        expectation = term_sum / n_mc

        # u = -1/λ * ln(expectation)
        u_val = -(1.0 / self.lambda_val) * torch.log(expectation)

        return u_val

    def has_exact_solution(self) -> bool:
        return True
