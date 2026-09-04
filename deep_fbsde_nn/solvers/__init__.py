"""
Deep BSDE Solvers
=================

Solvers:
- StandardSolver: Fixed X₀, one network u(t,x), Z by autograd (original repo design)
- GlobalSolver: Distributed X₀ (learns full solution surface)
- StepwiseSolver: Fixed X₀, trainable Y₀/Z₀ + per-timestep Z networks
  (Han-Jentzen-E 2018 original parameterization; strongest on nonlinear
  drivers like HJB, but learns u only at (0, X₀))
"""

from .base import BaseSolver, SolverConfig
from .global_solver import GlobalSolver
from .standard import StandardSolver
from .stepwise import StepwiseSolver

__all__ = [
    "BaseSolver",
    "SolverConfig",
    "StandardSolver",
    "GlobalSolver",
    "StepwiseSolver",
]
