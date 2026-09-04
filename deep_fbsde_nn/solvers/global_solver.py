"""
Global Deep BSDE Solver
=======================

Extension of Deep BSDE that learns the solution over a distribution of initial
conditions, not just a single fixed X0.

Key differences from Standard solver:
- X0 sampled from a distribution (e.g., uniform, log-normal)
- Network learns the full solution surface u(t, x)
- Better generalization to different initial conditions
- More useful for hedging applications

This approach is related to:
- Huré, Pham, Warin (2020): "Deep backward schemes for high-dimensional
  nonlinear PDEs"
- The "Global Deep BSDE" concept from various papers

Sampling strategies for X0:
- 'fixed': Same X0 for all paths (reduces to StandardSolver)
- 'uniform': Uniform in [low, high]^D
- 'lognormal': Log-normal around a center point
- 'gaussian': Gaussian around a center point
"""

from typing import Optional, Tuple, Union

import torch

from ..equations.base import BaseEquation
from .base import BaseSolver, SolverConfig


class GlobalSolver(BaseSolver):
    """
    Global Deep BSDE solver with distributed initial conditions.

    Args:
        equation: The PDE/BSDE equation
        network: Neural network u_θ(t, x)
        config: Solver configuration
        device: Compute device
        sampling: Initial condition sampling strategy
        X0_center: Center point for sampling (default: equation default)
        X0_spread: Spread/scale for sampling (default: 0.2)

    Example:
        >>> eq = BlackScholesEquation(dimension=100, strike=100)
        >>> net = NAISNet(input_dim=101, hidden_dim=256, output_dim=1)
        >>> solver = GlobalSolver(eq, net, SolverConfig(),
        ...                       sampling='lognormal', X0_center=100, X0_spread=0.2)
        >>> solver.train(n_iter=5000)
    """

    def __init__(
        self,
        equation: BaseEquation,
        network,
        config: SolverConfig,
        device="auto",
        sampling: str = "lognormal",
        X0_center: Optional[Union[float, torch.Tensor]] = None,
        X0_spread: float = 0.2,
        terminal_weight: float = 1.0,
    ):
        super().__init__(equation, network, config, device)

        self.sampling = sampling
        self.X0_spread = X0_spread
        self.terminal_weight = terminal_weight

        # Set center point
        if X0_center is None:
            self.X0_center = equation.sample_initial_condition(1).to(self.device)
        elif isinstance(X0_center, (int, float)):
            self.X0_center = torch.full(
                (1, equation.D), float(X0_center), device=self.device
            )
        else:
            self.X0_center = X0_center.to(self.device)
            if self.X0_center.dim() == 1:
                self.X0_center = self.X0_center.unsqueeze(0)

        print(f"Global solver with '{sampling}' sampling")
        print(f"  X0 center: {self.X0_center[0, 0].item():.2f}")
        print(f"  X0 spread: {X0_spread}")

    def _get_initial_condition(self, batch_size: int) -> torch.Tensor:
        """Sample X0 from the specified distribution."""
        D = self.equation.D
        center = self.X0_center.expand(batch_size, D)

        if self.sampling == "fixed":
            return center.clone()

        elif self.sampling == "uniform":
            # Uniform in [center*(1-spread), center*(1+spread)]
            low = center * (1 - self.X0_spread)
            high = center * (1 + self.X0_spread)
            return low + (high - low) * torch.rand(batch_size, D, device=self.device)

        elif self.sampling == "lognormal":
            # Log-normal: X = center * exp(σZ - σ²/2) where Z ~ N(0,1)
            Z = torch.randn(batch_size, D, device=self.device)
            sigma = self.X0_spread
            return center * torch.exp(sigma * Z - 0.5 * sigma**2)

        elif self.sampling == "gaussian":
            # Gaussian: X = center + spread * center * Z
            Z = torch.randn(batch_size, D, device=self.device)
            return center + self.X0_spread * center * Z

        else:
            raise ValueError(f"Unknown sampling: {self.sampling}")

    def _compute_loss(
        self, t: torch.Tensor, dW: torch.Tensor, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Global Deep BSDE loss.

        Same as standard, but with sampled X₀.
        """
        M = dW.shape[0]
        N = dW.shape[1]
        dt = self.equation.T / N

        # Sample initial conditions (key difference from Standard)
        X = self._get_initial_condition(M)

        # Get initial Y and Z
        t0 = t[0].view(1, 1).expand(M, 1)
        Y, Z = self.net_u(t0, X)
        Y0 = Y.mean()

        loss = torch.zeros(1, device=self.device)

        for n in range(N):
            t_n = t[n]
            dW_n = dW[:, n, :]

            mu = self.equation.drift(t_n, X, Y, Z)
            sigma = self.equation.diffusion(t_n, X, Y)
            f = self.equation.driver(t_n, X, Y, Z)

            # Forward SDE
            X_new = X + mu * dt + sigma * dW_n

            # BSDE prediction
            Y_pred = Y - f * dt + torch.sum(Z * sigma * dW_n, dim=1, keepdim=True)

            # Next step
            X = X_new.detach().requires_grad_(True)
            t_next = t[n + 1].view(1, 1).expand(M, 1)
            Y, Z = self.net_u(t_next, X)

            # Dynamics loss
            loss = loss + torch.mean((Y - Y_pred) ** 2)

        # Terminal losses
        g = self.equation.terminal(X)
        grad_g = self.equation.terminal_gradient(X)
        sigma_T = self.equation.diffusion(t[-1], X, Y)

        terminal_loss = torch.mean((Y - g) ** 2)
        Z_target = grad_g * sigma_T
        gradient_loss = torch.mean(torch.sum((Z - Z_target) ** 2, dim=1))

        loss = loss + self.terminal_weight * (terminal_loss + gradient_loss)

        return loss, Y0

    def price_surface(self, t: float, X_grid: torch.Tensor) -> torch.Tensor:
        """
        Evaluate price on a grid of X values.

        Args:
            t: Time
            X_grid: (num_points, D) grid of evaluation points

        Returns:
            (num_points, 1) prices
        """
        return self.predict(t, X_grid)

    def validate_at_points(
        self, t: float = 0.0, X_points: Optional[torch.Tensor] = None, n_points: int = 5
    ) -> dict:
        """
        Validate at multiple X points.

        Returns dict with predictions vs exact (if available).
        """
        self.model.eval()

        if X_points is None:
            # Generate test points around center
            X_points = self._get_initial_condition(n_points)

        predictions = self.predict(t, X_points)

        result = {
            "X": X_points.cpu().numpy(),
            "predictions": predictions.cpu().numpy(),
        }

        if self.equation.has_exact_solution():
            exact = self.equation.exact_solution(t, X_points)
            errors = torch.abs(predictions - exact) / torch.abs(exact) * 100

            result["exact"] = exact.cpu().numpy()
            result["relative_errors"] = errors.cpu().numpy()
            result["mean_error"] = errors.mean().item()

            print(f"Validation at t={t}:")
            print(f"  Mean relative error: {result['mean_error']:.2f}%")

        return result
