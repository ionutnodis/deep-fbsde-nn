"""
Deep FBSDE NN: Neural Networks for High-Dimensional PDEs
=========================================================

A PyTorch library for solving high-dimensional parabolic PDEs and
forward-backward stochastic differential equations (FBSDEs) with the
Deep BSDE method.

Features:
- NAIS-Net architecture with spectrally projected residual blocks
- Standard Deep BSDE (fixed initial condition)
- Global Deep BSDE (distributed initial conditions)
- Black-Scholes, Black-Scholes-Barenblatt, Allen-Cahn, and HJB equations

References:
- Han, Jentzen, E (2018): "Solving high-dimensional PDEs using deep learning"
- Ciccone et al. (2018): "NAIS-Net: Stable Deep Networks from Non-Autonomous DEs"
- Güler, Laignelet, Parpas (2019): "Towards robust and stable deep learning for FBSDEs"
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("deep-fbsde-nn")
except PackageNotFoundError:  # source checkout without `pip install -e .`
    __version__ = "0.0.0+unknown"

__author__ = "Ionut Nodis"

from . import equations, networks, solvers, utils

__all__ = [
    "networks",
    "equations",
    "solvers",
    "utils",
    "__version__",
]
