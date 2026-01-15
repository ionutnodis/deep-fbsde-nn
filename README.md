<h1 align="center">Deep FBSDE Neural Networks</h1>

A PyTorch library for solving high-dimensional Partial Differential Equations (PDEs) and Forward-Backward Stochastic Differential Equations (FBSDEs) using deep learning methods.

## Acknowledgements

This library was developed by **Ionut Nodis** as part of the **CQF (Certificate in Quantitative Finance) Final Project** (January 2026). The implementation builds upon example code and guidance provided by **Professor Panos Parpas** from Imperial College London.

The theoretical foundations are based on the Deep BSDE method introduced by Han, Jentzen, and E (2018), with neural network stability enhancements from the NAIS-Net architecture (Ciccone et al., 2018; Güler, Laignelet, Parpas, 2019).

## Overview

The Deep BSDE method reformulates high-dimensional PDEs as Forward-Backward SDEs and uses neural networks to approximate the solution. This approach overcomes the curse of dimensionality that affects traditional numerical methods, enabling the solution of PDEs in hundreds of dimensions.

### The FBSDE System

The PDE solution $u(t, x)$ corresponds to the backward component $Y_t$ of a coupled Forward-Backward SDE system:

$$
\begin{aligned}
\text{Forward SDE:} \quad & dX_t = \mu(t, X_t, Y_t, Z_t) \, dt + \sigma(t, X_t, Y_t) \, dW_t \\[6pt]
\text{Backward SDE:} \quad & dY_t = -f(t, X_t, Y_t, Z_t) \, dt + Z_t^\top \sigma(t, X_t, Y_t) \, dW_t \\[6pt]
\text{Terminal condition:} \quad & Y_T = g(X_T)
\end{aligned}
$$

where:
- $X_t \in \mathbb{R}^d$ is the forward process (e.g., asset prices)
- $Y_t \in \mathbb{R}$ is the backward process (the PDE solution)
- $Z_t \in \mathbb{R}^d$ is the gradient process
- $W_t$ is a $d$-dimensional Brownian motion

### Connection to PDEs (Feynman-Kac)

The FBSDE solution satisfies $Y_t = u(t, X_t)$ where $u$ solves the semilinear parabolic PDE:

$$
\frac{\partial u}{\partial t} + \mu \cdot \nabla u + \frac{1}{2} \text{Tr}\left(\sigma \sigma^\top D^2 u\right) + f(t, x, u, \sigma^\top \nabla u) = 0
$$

with terminal condition $u(T, x) = g(x)$.

The gradient relationship is $Z_t = \sigma^\top \nabla u(t, X_t)$.

## Features

- **High-dimensional PDEs**: Solve problems in $d = 10, 50, 100, 200+$ dimensions
- **NAIS-Net architecture**: Spectral projection ensures stability during training
- **Multiple equation types**: Black-Scholes, Black-Scholes-Barenblatt, Hamilton-Jacobi-Bellman
- **MLMC training**: Multi-Level Monte Carlo progressive time-stepping for faster convergence
- **Device agnostic**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU
- **Modular design**: Easy to extend with custom equations and architectures

## Installation

### From Git Repository

```bash
pip install git+https://github.com/ionutnodis/deep-fbsde-nn.git
```

### For Development

```bash
git clone https://github.com/ionutnodis/deep-fbsde-nn.git
cd deep-fbsde-nn
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.12
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- pandas >= 2.0.0
- tqdm >= 4.65.0

## Quick Start

```python
import torch
from deep_fbsde_nn.equations import BlackScholesBarenblattEquation, EquationConfig
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import StandardSolver, SolverConfig
from deep_fbsde_nn.utils import get_device

# Setup
device = get_device()  # Auto-detects CUDA/MPS/CPU
dimension = 100

# Define the equation (BSB has an analytical solution for validation)
equation = BlackScholesBarenblattEquation(
    dimension=dimension,
    sigma_min=0.1,
    sigma_max=0.3,
    terminal_time=1.0,
    device=device,
)

# Create the neural network
network = NAISNet(
    input_dim=dimension + 1,  # (t, X)
    hidden_dim=256,
    output_dim=1,
    num_layers=4,
    activation="sine",
)

# Configure the solver
solver_config = SolverConfig(
    batch_size=1,
    num_timesteps=50,
    learning_rate=1e-3,
    num_iterations=5000,
    use_mlmc=True,
)

# Create and train the solver
solver = StandardSolver(
    equation=equation,
    network=network,
    config=solver_config,
    device=device,
)

# Train
solver.train()

# Validate against exact solution
results = solver.validate()
print(f"Relative error: {results['relative_error']:.2f}%")
```

## Project Structure

```
deep_fbsde_nn/
├── deep_fbsde_nn/           # Main package
│   ├── __init__.py
│   ├── equations/           # PDE/BSDE equation definitions
│   │   ├── base.py          # BaseEquation abstract class
│   │   ├── black_scholes.py # Linear Black-Scholes (basket options)
│   │   ├── black_scholes_barenblatt.py  # Nonlinear with exact solution
│   │   └── hjb.py           # Hamilton-Jacobi-Bellman
│   ├── networks/            # Neural network architectures
│   │   ├── naisnet.py       # NAIS-Net (stable) & FeedForward
│   │   ├── activations.py   # Sine, ReLU, Tanh, etc.
│   │   └── wrapper.py       # Black-Scholes specific wrapper
│   ├── solvers/             # Deep BSDE solvers
│   │   ├── base.py          # BaseSolver with MLMC
│   │   ├── standard.py      # Fixed initial condition solver
│   │   └── global_solver.py # Distributed initial conditions
│   └── utils/               # Utilities
│       ├── device.py        # Device management
│       ├── metrics.py       # Error metrics, Monte Carlo
│       └── checkpointing.py # Model save/load
├── experiments/             # Experiment scripts
│   ├── exp01_bsb_dimension.py
│   ├── exp02_architecture_comparison.py
│   ├── exp03_black_scholes.py
│   ├── exp04_hjb.py
│   └── visualize.py
├── results/                 # Output directory
├── pyproject.toml           # Package configuration
└── README.md
```

## API Reference

### Equations

All equations inherit from `BaseEquation` and must implement:

| Method | Description |
|--------|-------------|
| `drift(t, X, Y, Z)` | Forward SDE drift $\mu(t, X, Y, Z)$ |
| `diffusion(t, X, Y)` | Forward SDE diffusion $\sigma(t, X, Y)$ |
| `driver(t, X, Y, Z)` | BSDE driver $f(t, X, Y, Z)$ |
| `terminal(X)` | Terminal condition $g(X)$ |
| `exact_solution(t, X)` | Analytical solution $u(t, X)$ (optional) |

**Available equations:**

| Equation | Description | Exact Solution |
|----------|-------------|----------------|
| `BlackScholesEquation` | Linear PDE for basket options | No (MC benchmark) |
| `BlackScholesBarenblattEquation` | Nonlinear with uncertain volatility | Yes |
| `HJBEquation` | Hamilton-Jacobi-Bellman control | No |
| `VanillaCallEquation` | Single-asset European call | Yes |

#### Black-Scholes-Barenblatt Equation

The BSB equation models option pricing under uncertain volatility $\sigma \in [\sigma_{\min}, \sigma_{\max}]$:

$$
\frac{\partial u}{\partial t} + \frac{1}{2} \bar{\sigma}^2(D^2 u) \|x\|^2 \, \text{Tr}(D^2 u) = 0
$$

where $\bar{\sigma}^2(D^2 u) = \sigma_{\max}^2 \mathbf{1}_{D^2 u \geq 0} + \sigma_{\min}^2 \mathbf{1}_{D^2 u < 0}$.

**Exact solution:** $u(t, x) = \|x\|^2 \exp\left(\sigma_{\max}^2 (T - t)\right)$

#### Hamilton-Jacobi-Bellman Equation

The HJB equation for stochastic optimal control:

$$
\frac{\partial u}{\partial t} + \Delta u - \lambda \|\nabla u\|^2 = 0
$$

### Networks

| Network | Description |
|---------|-------------|
| `NAISNet` | Stable architecture with spectral projection (recommended) |
| `FeedForwardNet` | Standard feedforward network (baseline) |

**NAIS-Net** enforces stability via spectral projection, ensuring each residual block is a contraction mapping. Given weight matrix $W$, the projection ensures $\|A\| < 1$ where:

$$
A = R^\top R + \varepsilon I, \quad \text{with } R^\top R = W^\top W \text{ rescaled if } \|W^\top W\| > \delta
$$

**NAISNet parameters:**

```python
NAISNet(
    input_dim=101,      # d + 1 (time + state)
    hidden_dim=256,     # Hidden layer width
    output_dim=1,       # Scalar output
    num_layers=4,       # Number of hidden layers
    epsilon=0.01,       # Stability parameter ε
    activation="sine",  # Activation function
)
```

### Solvers

| Solver | Description |
|--------|-------------|
| `StandardSolver` | Fixed initial condition $X_0$ (original Deep BSDE) |
| `GlobalSolver` | Distributed $X_0$ (learns full solution surface) |

The solver minimizes the loss function:

$$
\mathcal{L}(\theta) = \mathbb{E}\left[ \sum_{n=0}^{N-1} |Y_{n+1} - \hat{Y}_{n+1}|^2 + \lambda |Y_N - g(X_N)|^2 \right]
$$

where $\hat{Y}_{n+1} = Y_n - f(t_n, X_n, Y_n, Z_n) \Delta t + Z_n^\top \sigma \Delta W_n$.

**SolverConfig parameters:**

```python
SolverConfig(
    batch_size=1,           # Paths per iteration
    num_timesteps=50,       # Time discretization N
    learning_rate=1e-3,     # Adam learning rate
    num_iterations=20000,   # Training iterations
    use_mlmc=True,          # Multi-Level Monte Carlo
    gradient_clip=1.0,      # Gradient clipping
)
```

## Experiments

Run pre-configured experiments:

```bash
# BSB dimension scaling (d = 10, 50, 100, 200)
python experiments/exp01_bsb_dimension.py --all

# Architecture comparison (NAIS-Net vs FeedForward)
python experiments/exp02_architecture_comparison.py --dim 100

# Black-Scholes basket options
python experiments/exp03_black_scholes.py --dim 50

# Hamilton-Jacobi-Bellman
python experiments/exp04_hjb.py --dim 100
```

## Extending the Library

### Custom Equation

To implement a custom equation, subclass `BaseEquation` and implement the required methods:

```python
import torch
from deep_fbsde_nn.equations import BaseEquation, EquationConfig

class MyEquation(BaseEquation):
    """
    Example: HJB-type equation
        ∂u/∂t + Δu - (1/2)||∇u||² = 0
        u(T, x) = ln(0.5(1 + ||x||²))
    """
    def __init__(self, dimension: int, device=None):
        config = EquationConfig(name="MyEq", dimension=dimension)
        super().__init__(config, device)

    def drift(self, t, X, Y, Z):
        return torch.zeros_like(X)  # dX = σ dW

    def diffusion(self, t, X, Y):
        return torch.ones_like(X)   # σ = I

    def driver(self, t, X, Y, Z):
        # f = -Δu + (1/2)||∇u||² → for BSDE: f = -(1/2)||Z||²
        return -0.5 * torch.sum(Z**2, dim=1, keepdim=True)

    def terminal(self, X):
        # g(x) = ln(0.5(1 + ||x||²))
        return torch.log(0.5 * (1 + torch.sum(X**2, dim=1, keepdim=True)))
```

## References

1. **Han, J., Jentzen, A., & E, W.** (2018). Solving high-dimensional partial differential equations using deep learning. *PNAS*, 115(34), 8505-8510.

2. **Ciccone, M., Gallieri, M., Masci, J., Osendorfer, C., & Gomez, F.** (2018). NAIS-Net: Stable Deep Networks from Non-Autonomous Differential Equations. *NeurIPS*.

3. **Güler, R.A., Laignelet, A., & Parpas, P.** (2019). Towards Robust and Stable Deep Learning Algorithms for Forward Backward Stochastic Differential Equations. *arXiv:1910.11623*.

4. **E, W., Han, J., & Jentzen, A.** (2017). Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations. *Communications in Mathematics and Statistics*, 5(4), 349-380.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{nodis2026deepfbsde,
  author = {Nodis, Ionut},
  title = {Deep FBSDE Neural Networks},
  year = {2026},
  url = {https://github.com/your-username/deep-fbsde-nn}
}
```
