"""
Standard Deep BSDE Solver
=========================

The original Deep BSDE method from Han, Jentzen, E (2018).

Key characteristics:
- Fixed initial condition X₀ (same for all paths in a batch)
- Network learns u(t, X) where Y_t = u(t, X_t)
- Loss penalizes deviation from BSDE dynamics and terminal condition

Algorithm:
    1. Fix X₀ (e.g., all assets at strike)
    2. Simulate forward SDE: X_{n+1} = X_n + μΔt + σΔW_n
    3. Compute Y_n = u_θ(t_n, X_n) and Z_n = ∇_X u_θ(t_n, X_n)
    4. Check BSDE dynamics: Y_{n+1} ≈ Y_n - f(...)Δt + Z_n·σΔW_n
    5. Terminal condition: Y_N ≈ g(X_N)

Loss function:
    L(θ) = E[ Σ_n |Y_{n+1} - Ŷ_{n+1}|² + λ|Y_N - g(X_N)|² + λ|Z_N - ∇g(X_N)|² ]

    where Ŷ_{n+1} = Y_n - f(t_n, X_n, Y_n, Z_n)Δt + Z_n · σ(t_n, X_n) · ΔW_n
"""

import torch
from typing import Tuple, Optional
from .base import BaseSolver, SolverConfig
from ..equations.base import BaseEquation


class StandardSolver(BaseSolver):
    """
    Standard Deep BSDE solver with fixed initial condition.
    """

    def __init__(
        self,
        equation: BaseEquation,
        network,
        config: SolverConfig,
        device="auto",
        X0: Optional[torch.Tensor] = None,
        terminal_weight: float = 1.0,
    ):
        super().__init__(equation, network, config, device)

        if X0 is None:
            self.X0 = equation.sample_initial_condition(1).to(self.device)
        else:
            self.X0 = X0.to(self.device)
            if self.X0.dim() == 1:
                self.X0 = self.X0.unsqueeze(0)
        self.terminal_weight = terminal_weight

    def _get_initial_condition(self, batch_size: int) -> torch.Tensor:
        return self.X0.expand(batch_size, -1).clone()

    def _compute_loss(
        self, t: torch.Tensor, dW: torch.Tensor, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M = dW.shape[0]
        N = dW.shape[1]
        dt = self.equation.T / N

        X = self._get_initial_condition(M)

        # Get initial Y and Z
        t0 = t[0]
        t0_batch = t0.view(1, 1).expand(M, 1)
        Y, Z = self.net_u(t0_batch, X)
        Y0 = Y[0, 0]

        loss = torch.zeros(1, device=self.device)
        W0 = W[:, 0, :]

        for n in range(N):
            t_n = t[n]
            t_n1 = t[n + 1]
            dt_n = t_n1 - t_n
            W1 = W[:, n + 1, :]
            dW_n = W1 - W0

            # --- CORRELATION FIX ---
            # If the equation has a Cholesky factor L, apply it to the noise.
            if (
                hasattr(self.equation, "L")
                and hasattr(self.equation, "has_correlation")
                and self.equation.has_correlation
            ):
                # dW_n is (M, D). L is (D, D). Result: (M, D)
                dW_corr = dW_n @ self.equation.L.t()
            else:
                dW_corr = dW_n
            # -----------------------

            mu = self.equation.drift(t_n, X, Y, Z)
            sigma = self.equation.diffusion(t_n, X, Y)
            phi = self.equation.driver(t_n, X, Y, Z)

            # Use dW_corr for state evolution
            X_new = X + mu * dt_n + sigma * dW_corr

            # Use dW_corr for BSDE consistency
            Y_pred = (
                Y + phi * dt_n + torch.sum(Z * sigma * dW_corr, dim=1, keepdim=True)
            )

            X = X_new.detach().requires_grad_(True)
            t_n1_batch = t_n1.view(1, 1).expand(M, 1)
            Y, Z = self.net_u(t_n1_batch, X)

            loss = loss + torch.sum((Y - Y_pred) ** 2)
            W0 = W1

        g = self.equation.terminal(X)
        loss = loss + torch.sum((Y - g) ** 2)

        return loss, Y0

    def get_price_and_delta(
        self, t: float = 0.0, X: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if X is None:
            X = self.X0.clone()
        self.model.eval()
        X = X.to(self.device).requires_grad_(True)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        t_tensor = torch.full((X.shape[0], 1), t, device=self.device)

        # Note: If network wrapper expects transformed inputs, ensure X is raw
        # The wrapper inside self.model should handle raw X -> log(X/K)
        u = self.model(torch.cat([t_tensor, X], dim=1))
        delta = torch.autograd.grad(u.sum(), X)[0]
        return u.detach(), delta.detach()

    def validate(self, n_samples: int = 1000) -> dict:
        """
        Validate against exact solution (if available).

        Returns:
            Dict with 'Y0_pred', 'Y0_exact', 'relative_error'
        """
        self.model.eval()

        # Predict at t=0, X=X₀
        Y0_pred = self.predict(0.0, self.X0).item()

        result = {"Y0_pred": Y0_pred}

        if self.equation.has_exact_solution():
            Y0_exact = self.equation.exact_solution(0.0, self.X0).item()
            rel_error = abs(Y0_pred - Y0_exact) / abs(Y0_exact) * 100

            result["Y0_exact"] = Y0_exact
            result["relative_error"] = rel_error

            print(f"Y0 predicted: {Y0_pred:.6f}")
            print(f"Y0 exact:     {Y0_exact:.6f}")
            print(f"Relative error: {rel_error:.2f}%")

        return result
