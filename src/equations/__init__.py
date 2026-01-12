"""
PDE/BSDE Equation Definitions
=============================

Equations for Deep BSDE solvers:
- BlackScholesEquation: Linear, basket options
- BlackScholesBarenblattEquation: Nonlinear, uncertain volatility (has exact solution)
- AllenCahnEquation: Semilinear, reaction-diffusion
"""

from .base import BaseEquation, EquationConfig
from .black_scholes import BlackScholesEquation, VanillaCallEquation
from .black_scholes_barenblatt import BlackScholesBarenblattEquation, AllenCahnEquation
from .hjb import HJBEquation

__all__ = [
    "BaseEquation",
    "EquationConfig",
    "BlackScholesEquation",
    "BlackScholesBarenblattEquation",
    "AllenCahnEquation",
    "VanillaCallEquation",
    "HJBEquation",
]
