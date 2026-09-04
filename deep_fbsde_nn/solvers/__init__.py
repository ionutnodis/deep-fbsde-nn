"""
Deep BSDE Solvers
=================

Solvers:
- StandardSolver: Fixed X₀ (original Deep BSDE)
- GlobalSolver: Distributed X₀ (learns full solution surface)
"""

from .base import BaseSolver, SolverConfig
from .global_solver import GlobalSolver
from .standard import StandardSolver

__all__ = [
    "BaseSolver",
    "SolverConfig",
    "StandardSolver",
    "GlobalSolver",
]
