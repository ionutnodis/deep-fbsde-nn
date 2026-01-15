"""
Deep BSDE Solvers
=================

Solvers:
- StandardSolver: Fixed X₀ (original Deep BSDE)
- GlobalSolver: Distributed X₀ (learns full solution surface)
"""

from .base import BaseSolver, SolverConfig
from .standard import StandardSolver
from .global_solver import GlobalSolver

__all__ = [
    "BaseSolver",
    "SolverConfig",
    "StandardSolver",
    "GlobalSolver",
]
