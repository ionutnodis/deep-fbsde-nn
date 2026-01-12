"""
Base Solver
===========

Abstract base class for Deep BSDE solvers.

Features:
- MLMC (Multi-Level Monte Carlo) progressive time-stepping
- Learning rate warmup + decay
- Gradient clipping
- Checkpointing

MLMC Schedule:
    Progress   Timesteps
    0-10%      N=5
    10-25%     N=10
    25-50%     N=20
    50-75%     N=30
    75-100%    N=50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import time

from ..equations.base import BaseEquation
from ..utils.device import get_device


@dataclass
class SolverConfig:
    """Solver configuration."""

    batch_size: int = 1  # Reference uses M=1
    num_timesteps: int = 50
    learning_rate: float = 1e-3
    num_iterations: int = 20000

    # MLMC - iteration-based like reference
    # Reference: N = ceil(Mm^(floor(it/4000)+1)) where Mm = N^0.2 ≈ 2.19
    # it < 4000: N ≈ 3
    # it 4000-8000: N ≈ 5
    # it 8000-12000: N ≈ 10
    # it 12000-16000: N ≈ 22
    # it 16000-20000: N ≈ 48
    use_mlmc: bool = True
    mlmc_schedule: Dict[int, int] = field(
        default_factory=lambda: {0: 5, 2000: 10, 5000: 50, 10000: 50, 16000: 50}
    )

    # Optimization
    gradient_clip: float = 1.0
    weight_decay: float = 0.0  # Reference doesn't use weight decay
    warmup_steps: int = 0  # Reference doesn't use warmup
    lr_decay: float = 1.0  # No decay in first phase
    lr_decay_at: float = 1.0

    # Logging
    print_every: int = 100


class BaseSolver(ABC):
    """
    Abstract base class for FBSDE solvers.

    Subclasses must implement:
    - _compute_loss(t, W): Compute loss from time grid and Brownian paths
    - _get_initial_condition(batch_size): Get X_0
    """

    def __init__(
        self,
        equation: BaseEquation,
        network: nn.Module,
        config: SolverConfig,
        device: Union[str, torch.device] = "auto",
    ):
        if isinstance(device, str):
            self.device = get_device(device)
        else:
            self.device = device

        self.equation = equation
        self.equation.device = self.device
        self.model = network.to(self.device)
        self.config = config

        # Training state
        self.training_history: Dict[str, List] = {
            "iterations": [],
            "losses": [],
            "Y0": [],
        }
        self.current_iteration = 0
        self._is_training = False

        self.optimizer: Optional[optim.Optimizer] = None

        print(
            f"Solver on {self.device} | {equation.name} | "
            f"{sum(p.numel() for p in network.parameters()):,} params"
        )

    def _get_mlmc_timesteps(self, iteration: int) -> int:
        """Get timesteps based on MLMC schedule (iteration-based)."""
        if not self.config.use_mlmc:
            return self.config.num_timesteps

        N = self.config.num_timesteps
        for thresh, steps in sorted(self.config.mlmc_schedule.items()):
            if iteration >= thresh:
                N = steps
        return N

    def generate_paths(
        self, M: int, N: int, L: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate time grid, Brownian increments, and paths.

        Returns:
            t: (N+1,) time grid
            dW: (M, N, D) Brownian increments
            W: (M, N+1, D) Brownian paths
        """
        T, D = self.equation.T, self.equation.D
        dt = T / N

        # Time grid
        t = torch.linspace(0, T, N + 1, device=self.device)

        # Brownian increments
        dW = np.sqrt(dt) * torch.randn(M, N, D, device=self.device)
        if L is not None:
            dW = dW @ L.T

        # Cumulative paths
        W = torch.zeros(M, N + 1, D, device=self.device)
        W[:, 1:, :] = torch.cumsum(dW, dim=1)

        return t, dW, W

    def net_u(
        self, t: torch.Tensor, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute u(t,X) and Z = ∇_X u.

        Args:
            t: (M, 1) or scalar
            X: (M, D)
        Returns:
            u: (M, 1)
            Z: (M, D)
        """
        if not X.requires_grad:
            X = X.detach().requires_grad_(True)

        # Handle time input
        if isinstance(t, (int, float)):
            t = torch.full((X.shape[0], 1), t, device=self.device)
        elif t.dim() == 0:
            t = t.view(1, 1).expand(X.shape[0], 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)

        # Forward
        inputs = torch.cat([t, X], dim=1)
        u = self.model(inputs)

        # Gradient
        ones = torch.ones_like(u)
        Z = torch.autograd.grad(
            u, X, ones, create_graph=self._is_training, retain_graph=True
        )[0]

        return u, Z

    @abstractmethod
    def _compute_loss(
        self, t: torch.Tensor, dW: torch.Tensor, W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss. Returns (loss, Y0)."""
        pass

    @abstractmethod
    def _get_initial_condition(self, batch_size: int) -> torch.Tensor:
        """Get X_0 for batch. Returns (batch_size, D)."""
        pass

    def train(
        self,
        n_iter: Optional[int] = None,
        learning_rate: Optional[float] = None,
        print_every: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Train the solver.

        Can be called multiple times for chunked training.
        Optimizer state is preserved between calls.
        """
        n_iter = n_iter or self.config.num_iterations
        learning_rate = learning_rate or self.config.learning_rate
        print_every = print_every or self.config.print_every

        self._is_training = True
        self.model.train()

        # Create optimizer only if not exists (allows resumable training)
        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )

        # Correlation matrix (if equation has one)
        L = getattr(self.equation, "L", None)

        start_time = time.time()
        loss_buffer = []

        start_iter = self.current_iteration
        end_iter = start_iter + n_iter

        for it in range(start_iter, end_iter):
            # Get MLMC timesteps based on iteration count
            N = self._get_mlmc_timesteps(it)

            # Generate paths
            t, dW, W = self.generate_paths(self.config.batch_size, N, L)

            # Forward + backward
            self.optimizer.zero_grad(set_to_none=True)
            loss, Y0 = self._compute_loss(t, dW, W)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()

            loss_buffer.append(loss.item())
            Y0_val = Y0.item() if isinstance(Y0, torch.Tensor) else Y0

            # Log
            if it % print_every == 0 or it == end_iter - 1:
                avg_loss = np.mean(loss_buffer) if loss_buffer else loss.item()
                elapsed = time.time() - start_time

                print(
                    f"It {it:5d} | Loss {avg_loss:.3e} | Y0 {Y0_val:.4f} | "
                    f"N {N:2d} | {elapsed:.1f}s"
                )

                self.training_history["iterations"].append(it)
                self.training_history["losses"].append(avg_loss)
                self.training_history["Y0"].append(Y0_val)
                loss_buffer = []
                start_time = time.time()

        self.current_iteration = end_iter
        self._is_training = False
        self.model.eval()

        return self.training_history

    def predict(self, t: float, X: torch.Tensor) -> torch.Tensor:
        """Predict u(t, X)."""
        self.model.eval()
        with torch.no_grad():
            if X.dim() == 1:
                X = X.unsqueeze(0)
            X = X.to(self.device)
            t_tensor = torch.full((X.shape[0], 1), t, device=self.device)
            return self.model(torch.cat([t_tensor, X], dim=1))

    def save(self, path: str):
        """Save checkpoint."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "history": self.training_history,
                "iteration": self.current_iteration,
            },
            path,
        )

    def load(self, path: str):
        """Load checkpoint.

        For PyTorch 2.6+, we need to allowlist numpy scalar types
        since training history may contain numpy arrays.
        """
        import numpy as np

        # Add numpy types to safe globals for weights_only=True loading
        try:
            # For numpy >= 2.0
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        except AttributeError:
            # For numpy < 2.0
            try:
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            except AttributeError:
                pass

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.training_history = ckpt.get("history", {})
        self.current_iteration = ckpt.get("iteration", 0)
