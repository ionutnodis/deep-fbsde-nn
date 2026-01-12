"""
Base Equation Class
===================

Abstract base class for PDE/BSDE equations.

The FBSDE system:
    Forward SDE:   dX_t = μ(t,X,Y,Z)dt + σ(t,X,Y)dW_t
    Backward SDE:  dY_t = -f(t,X,Y,Z)dt + Z_t·σ dW_t
    Terminal:      Y_T = g(X_T)

Connection to PDE (Feynman-Kac):
    Y_t = u(t, X_t)  where u solves
    ∂u/∂t + μ·∇u + (1/2)Tr(σσᵀ D²u) + f(t,x,u,σᵀ∇u) = 0
    u(T, x) = g(x)
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class EquationConfig:
    """Configuration for a PDE/BSDE equation."""

    name: str
    dimension: int
    terminal_time: float = 1.0


class BaseEquation(ABC):
    """
    Abstract base class for FBSDE equations.

    Subclasses must implement:
    - drift(): μ(t,X,Y,Z)
    - diffusion(): σ(t,X,Y)
    - driver(): f(t,X,Y,Z)
    - terminal(): g(X)

    Optional:
    - terminal_gradient(): ∇g(X) - defaults to autodiff
    - exact_solution(): u(t,X) if known analytically
    """

    def __init__(self, config: EquationConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cpu")

        # Shortcuts
        self.D = config.dimension
        self.T = config.terminal_time
        self.name = config.name

    @abstractmethod
    def drift(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward SDE drift μ(t,X,Y,Z).

        Args:
            t: (M, 1), X: (M, D), Y: (M, 1), Z: (M, D)
        Returns:
            (M, D)
        """
        pass

    @abstractmethod
    def diffusion(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward SDE diffusion σ(t,X,Y).

        Returns:
            (M, D) for diagonal diffusion, or (M, D, D) for full matrix
        """
        pass

    @abstractmethod
    def driver(
        self, t: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """
        BSDE driver f(t,X,Y,Z).

        Note: The solver uses dY = -f dt + Z·σ dW, so return f (not -f).

        Returns:
            (M, 1)
        """
        pass

    @abstractmethod
    def terminal(self, X: torch.Tensor) -> torch.Tensor:
        """
        Terminal condition g(X).

        Args:
            X: (M, D)
        Returns:
            (M, 1)
        """
        pass

    def terminal_gradient(self, X: torch.Tensor) -> torch.Tensor:
        """
        Gradient ∇g(X). Default uses autodiff; override for speed.

        Returns:
            (M, D)
        """
        X_grad = X.detach().requires_grad_(True)
        g = self.terminal(X_grad)

        grad = torch.autograd.grad(
            outputs=g,
            inputs=X_grad,
            grad_outputs=torch.ones_like(g),
            create_graph=False,
        )[0]

        return grad

    def exact_solution(
        self, t: Union[float, torch.Tensor], X: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Analytical solution u(t,X) if available. Returns None by default."""
        return None

    def exact_gradient(
        self, t: Union[float, torch.Tensor], X: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Analytical gradient ∇u(t,X) if available."""
        return None

    def sample_initial_condition(self, batch_size: int = 1) -> torch.Tensor:
        """Sample initial conditions X_0. Override for equation-specific sampling."""
        return torch.ones(batch_size, self.D, device=self.device)

    def has_exact_solution(self) -> bool:
        """Check if analytical solution is available."""
        try:
            result = self.exact_solution(0.0, torch.ones(1, self.D, device=self.device))
            return result is not None
        except:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(D={self.D}, T={self.T})"
