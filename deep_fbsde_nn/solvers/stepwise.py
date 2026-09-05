"""
Stepwise Deep BSDE Solver — the original Han-Jentzen-E parameterization.

Instead of one network u(t, x) with Z = ∇u by autograd (StandardSolver),
this solver treats the initial value Y₀ and initial gradient Z₀ as trainable
parameters and learns a SEPARATE small network Z_n = net_n(X_n) for each
interior timestep. Only the terminal condition is penalized.

Why it exists: deriving Z from the same network as u lets the optimizer damp
the martingale term by under-estimating ||Z||, which under-weights strongly
nonlinear (quadratic-in-Z) drivers — StandardSolver plateaus ~12-15% above
the HJB reference regardless of budget. Decoupling Z per timestep removes
that failure mode at the cost of learning the solution only at (0, X₀).

Trade-offs vs StandardSolver:
- Recovers u(0, X₀) and Z(0, X₀) only — no solution surface, no greeks
  away from X₀. Use StandardSolver/GlobalSolver when you need u(t, x).
- The time grid is FIXED at construction (one net per step): MLMC
  progressive time-stepping is not supported.

Reference:
    Han, Jentzen, E (2018): "Solving high-dimensional partial differential
    equations using deep learning", PNAS 115(34) — the "Deep BSDE" method.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..equations.base import BaseEquation
from ..networks import FeedForwardNet
from .base import BaseSolver, SolverConfig


class _StepwiseModel(nn.Module):
    """Container so BaseSolver's optimizer/checkpoint machinery sees one module.

    Holds:
        Y0     — trainable scalar, the estimate of u(0, X0)
        Z0     — trainable (1, D), the estimate of Z at t=0
        z_nets — ModuleList of N-1 networks, Z_n = z_nets[n-1](X_n)
    """

    def __init__(
        self,
        dimension: int,
        num_timesteps: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        network_cls=FeedForwardNet,
    ):
        super().__init__()
        self.Y0 = nn.Parameter(torch.zeros(1, 1))
        self.Z0 = nn.Parameter(torch.zeros(1, dimension))
        self.z_nets = nn.ModuleList(
            network_cls(
                input_dim=dimension,
                hidden_dim=hidden_dim,
                output_dim=dimension,
                num_layers=num_layers,
                activation=activation,
            )
            for _ in range(num_timesteps - 1)
        )
        # Zero the output layers so every net starts at Z ≡ 0. With random
        # init, ||Z||² scales like the dimension and injects a huge spurious
        # drift through nonlinear drivers (observed at d=100: initial loss
        # ~400 and Y0 dragged out of the correct basin). Starting at Z=0 makes
        # the initial dynamics exactly consistent with the warm-started Y0.
        for net in self.z_nets:
            output_layer = getattr(net, "output_layer", None)
            if output_layer is not None:
                nn.init.zeros_(output_layer.weight)
                if output_layer.bias is not None:
                    nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise RuntimeError(
            "_StepwiseModel is not a plain network; use StepwiseSolver.predict()."
        )


class StepwiseSolver(BaseSolver):
    """
    Deep BSDE solver with per-timestep Z networks (Han et al. 2018 design).

    Args:
        equation: The PDE/BSDE equation
        config: Solver configuration. ``use_mlmc`` must be False — the time
            grid is baked into the per-step networks.
        device: Compute device
        X0: Fixed initial condition (defaults to equation's)
        hidden_dim / num_layers / activation: architecture of each Z net
        network_cls: class for the Z nets (FeedForwardNet default; NAISNet works)

    Training recipe: at low dimension (d ≤ ~10) batch 64 with lr 5e-3 works;
    at high dimension use a LARGER batch and SMALLER learning rate — noisy Z
    networks feed spurious drift through nonlinear drivers and Y0 chases it.
    Reference point: HJB d=100 with batch 256, lr 5e-4, 3000 iterations
    reproduces the published Han et al. value 4.5901 to 0.03% on CPU (~70s).

    Example:
        >>> eq = HJBEquation(dimension=3)
        >>> config = SolverConfig(batch_size=64, num_timesteps=20,
        ...                       num_iterations=3000, use_mlmc=False)
        >>> solver = StepwiseSolver(eq, config, device="cpu")
        >>> solver.train()
        >>> y0 = solver.predict()          # u(0, X0)
    """

    def __init__(
        self,
        equation: BaseEquation,
        config: SolverConfig,
        device="auto",
        X0: Optional[torch.Tensor] = None,
        hidden_dim: int = 32,
        num_layers: int = 3,
        activation: str = "sine",
        network_cls=FeedForwardNet,
    ):
        if config.use_mlmc:
            raise ValueError(
                "StepwiseSolver has one Z network per timestep; the grid cannot "
                "change during training. Set SolverConfig(use_mlmc=False)."
            )
        if getattr(equation, "has_correlation", False):
            raise NotImplementedError(
                "StepwiseSolver does not support correlated Brownian motion yet; "
                "use StandardSolver for correlated baskets."
            )

        model = _StepwiseModel(
            dimension=equation.D,
            num_timesteps=config.num_timesteps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            network_cls=network_cls,
        )
        super().__init__(equation, model, config, device)

        if X0 is None:
            self.X0 = equation.sample_initial_condition(1).to(self.device)
        else:
            self.X0 = X0.to(self.device)
            if self.X0.dim() == 1:
                self.X0 = self.X0.unsqueeze(0)

        self._warm_start_y0()

    def _warm_start_y0(self, n_paths: int = 512) -> None:
        """Initialize Y0 at the driver-free value E[g(X_T)].

        With a one-signed driver (e.g. HJB's +λ||Z||² drift) the loss has
        mirror basins: from Y0=0 at high dimension the optimizer can converge
        to ≈ -u(0,X0) (observed at d=100: -4.64 vs +4.59). Han et al. avoid
        this with a hand-tuned per-problem ``y_init_range``; simulating the
        forward SDE once and starting at the linear (driver-free)
        approximation achieves the same basin selection with no tuning.
        Draws from the global RNG — seed before construction for
        reproducibility.
        """
        with torch.no_grad():
            N = self.config.num_timesteps
            dt = self.equation.T / N
            X = self._get_initial_condition(n_paths)
            Y = torch.zeros(n_paths, 1, device=self.device)
            Z = torch.zeros(n_paths, self.equation.D, device=self.device)
            for n in range(N):
                t_n = torch.tensor(n * dt, device=self.device)
                mu = self.equation.drift(t_n, X, Y, Z)
                sigma = self.equation.diffusion(t_n, X, Y)
                dW = np.sqrt(dt) * torch.randn(
                    n_paths, self.equation.D, device=self.device
                )
                X = X + mu * dt + sigma * dW
            self.model.Y0.data.fill_(self.equation.terminal(X).mean().item())

    def _get_initial_condition(self, batch_size: int) -> torch.Tensor:
        return self.X0.expand(batch_size, -1).clone()

    def _compute_loss(
        self, t: torch.Tensor, dW: torch.Tensor, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M, N = dW.shape[0], dW.shape[1]
        if N != self.config.num_timesteps:  # defensive: grid must match the nets
            raise RuntimeError(
                f"Time grid N={N} does not match the {self.config.num_timesteps} "
                "steps this solver was built for."
            )

        X = self._get_initial_condition(M)
        Y = self.model.Y0.expand(M, 1)
        Z = self.model.Z0.expand(M, self.equation.D)

        for n in range(N):
            t_n = t[n]
            dt_n = t[n + 1] - t_n
            dW_n = dW[:, n, :]

            mu = self.equation.drift(t_n, X, Y, Z)
            sigma = self.equation.diffusion(t_n, X, Y)
            f = self.equation.driver(t_n, X, Y, Z)

            # Documented convention: dY = -f dt + Z^T sigma dW
            Y = Y - f * dt_n + torch.sum(Z * sigma * dW_n, dim=1, keepdim=True)
            X = X + mu * dt_n + sigma * dW_n

            if n < N - 1:
                Z = self.model.z_nets[n](X)

        g = self.equation.terminal(X)
        loss = torch.mean((Y - g) ** 2)

        return loss, self.model.Y0.detach().view(-1)[0]

    def predict(self, t: float = 0.0, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the learned u(0, X0).

        This solver learns the solution only at the initial point; asking for
        any other (t, X) is a usage error, reported loudly rather than
        answered wrongly.
        """
        if t != 0.0:
            raise ValueError(
                "StepwiseSolver only learns u at t=0 (Han et al. design); "
                "use StandardSolver or GlobalSolver for u(t, x) surfaces."
            )
        if X is not None:
            X = X.to(self.device)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            if not torch.allclose(X, self.X0.expand_as(X)):
                raise ValueError(
                    "StepwiseSolver only learns u at its fixed X0; got a "
                    "different evaluation point."
                )
        return self.model.Y0.detach().clone()

    def delta0(self) -> torch.Tensor:
        """The learned gradient Z0 = ∇u(0, X0) — the delta at the anchor.

        Pointwise greeks recipe: anchor a StepwiseSolver at each spot you
        care about (equation X0 = query point, same strike) and read
        ``predict()`` for the price and ``delta0()`` for the delta. Each
        anchored run is its own well-scaled problem, so far-from-the-money
        points are as accurate as at-the-money ones (validated for
        XVAEquation across S = 75..130: worst delta error ~0.01).
        """
        return self.model.Z0.detach().clone()

    def validate(self) -> dict:
        """Compare the learned Y0 against the equation's reference, if any."""
        result = {"Y0_pred": self.model.Y0.item()}
        if self.equation.has_exact_solution():
            exact = self.equation.exact_solution(0.0, self.X0)
            if exact is not None:
                y0_exact = exact.item() if torch.is_tensor(exact) else float(exact)
                result["Y0_exact"] = y0_exact
                result["relative_error"] = (
                    abs(result["Y0_pred"] - y0_exact) / abs(y0_exact) * 100
                )
        return result

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"StepwiseSolver({self.equation.name}, D={self.equation.D}, "
            f"N={self.config.num_timesteps}, params={n_params:,})"
        )


__all__ = ["StepwiseSolver", "SolverConfig"]
