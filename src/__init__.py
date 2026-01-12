"""
Deep BSDE: Neural Networks for High-Dimensional PDEs
=====================================================

A PyTorch implementation of Deep BSDE methods for solving
high-dimensional parabolic PDEs in quantitative finance.

Features:
- NAIS-Net architecture with stability guarantees
- Standard Deep BSDE (fixed initial condition)
- Global Deep BSDE (distributed initial conditions)
- Black-Scholes-Merton and Black-Scholes-Barenblatt equations

References:
- Han, Jentzen, E (2018): "Solving high-dimensional PDEs using deep learning"
- Ciccone et al. (2018): "NAIS-Net: Stable Deep Networks from Non-Autonomous DEs"
- Güler, Laignelet, Parpas (2019): "Towards robust and stable deep learning for FBSDEs"

Author: Ionut Nodis
Project: CQF Final Project (January 2026)
"""

__version__ = "1.0.0"
__author__ = "Ionut Nodis"

from src import networks
from src import equations
from src import solvers
from src import utils
