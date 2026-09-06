# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Write your own equation in ~30 lines
#
# The library ships six validated equations — but the point of the
# `BaseEquation` API is that *your* model is a small subclass away. This
# notebook adds something the library doesn't have: a European call on a
# **dividend-paying** stock, and validates the solver against Merton's
# closed form. If you're evaluating whether to build on this package,
# this is the API you'd actually live with.
#
# Generated from
# [`examples/custom_equation_notebook.py`](https://github.com/ionutnodis/deep-fbsde-nn/blob/main/examples/custom_equation_notebook.py)
# (jupytext percent format) and **executed in CI** (at the `NB_FAST=1`
# setting), so it cannot silently rot.

# %%
try:
    import deep_fbsde_nn
except ImportError:  # e.g. on Colab
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "deep-fbsde-nn"], check=True)
    import deep_fbsde_nn

print("deep-fbsde-nn", deep_fbsde_nn.__version__)

# %%
import math
import os

import torch

from deep_fbsde_nn.equations import EquationConfig
from deep_fbsde_nn.equations.base import BaseEquation
from deep_fbsde_nn.solvers import SolverConfig, StepwiseSolver

# CI executes this notebook with NB_FAST=1 (fewer iterations).
FAST = os.environ.get("NB_FAST") == "1"
ITERS, FINE_ITERS = (400, 200) if FAST else (2000, 1000)
torch.manual_seed(0)

# %% [markdown]
# ## The contract
#
# A `BaseEquation` subclass describes the FBSDE system
#
# $$dX_t = \mu\,dt + \sigma\,dW_t, \qquad
#   dY_t = -f\,dt + Z^\top \sigma\,dW_t, \qquad Y_T = g(X_T)$$
#
# via four required methods — `drift` (μ), `diffusion` (σ), `driver` (f),
# `terminal` (g) — plus optional extras (`exact_solution` for validation,
# `terminal_gradient` for speed, `sample_initial_condition` for the
# anchor point). Note the sign convention: `driver` returns $f$, and the
# solver integrates $dY = -f\,dt + \dots$
#
# For a call on a stock paying a continuous dividend yield $q$: the
# risk-neutral drift drops to $(r - q)X$, discounting stays at $r$
# (driver $-rY$), and Merton (1973) gives the closed form to grade
# against.


# %%
def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class DividendCallEquation(BaseEquation):
    """European call on a stock with a continuous dividend yield q."""

    def __init__(self, S0=100.0, strike=100.0, r=0.05, q=0.03, sigma=0.2,
                 terminal_time=1.0, device=None):
        config = EquationConfig(name="Dividend call", dimension=1,
                                terminal_time=terminal_time)
        super().__init__(config, device)
        self.S0, self.strike = S0, strike
        self.r, self.q, self.sigma = r, q, sigma

    # --- the four required methods -----------------------------------
    def drift(self, t, X, Y, Z):
        return (self.r - self.q) * X          # dividends leak out of the drift

    def diffusion(self, t, X, Y):
        return self.sigma * X

    def driver(self, t, X, Y, Z):
        return -self.r * Y                    # discounting (solver does dY = -f dt + ...)

    def terminal(self, X):
        return torch.relu(X[:, :1] - self.strike)

    # --- optional, but they buy you validation and speed -------------
    def terminal_gradient(self, X):
        return (X > self.strike).float()

    def sample_initial_condition(self, batch_size=1):
        return torch.full((batch_size, self.D), self.S0, device=self.device)

    def exact_solution(self, t, X):
        """Merton (1973): S e^{-qτ} N(d1) - K e^{-rτ} N(d2)."""
        tau = self.T - float(t)
        S = X[:, :1]
        if tau <= 0:
            return torch.relu(S - self.strike)
        d1 = (torch.log(S / self.strike)
              + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * math.sqrt(tau))
        d2 = d1 - self.sigma * math.sqrt(tau)
        return (S * math.exp(-self.q * tau) * _norm_cdf(d1)
                - self.strike * math.exp(-self.r * tau) * _norm_cdf(d2))

    def exact_gradient(self, t, X):
        """Merton delta: e^{-qτ} N(d1)."""
        tau = self.T - float(t)
        S = X[:, :1]
        if tau <= 0:
            return (S > self.strike).float()
        d1 = (torch.log(S / self.strike)
              + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (self.sigma * math.sqrt(tau))
        return math.exp(-self.q * tau) * _norm_cdf(d1)


# %% [markdown]
# ## Train and validate
#
# That's the whole integration — every solver in the library now accepts
# the new equation. The dividend yield should knock a visible chunk off
# the no-dividend price (q=3% on an ATM call is not subtle).

# %%
equation = DividendCallEquation()
config = SolverConfig(
    batch_size=64, num_timesteps=20, learning_rate=5e-3,
    num_iterations=ITERS, use_mlmc=False, print_every=10**9,
)
solver = StepwiseSolver(equation, config, device="cpu")
solver.train()
if FINE_ITERS:
    for group in solver.optimizer.param_groups:
        group["lr"] = 1e-3
    solver.train(n_iter=FINE_ITERS)

result = solver.validate()
print(f"solver price:   {result['Y0_pred']:.4f}")
print(f"Merton price:   {result['Y0_exact']:.4f}")
print(f"relative error: {result['relative_error']:.2f}%")

no_div = DividendCallEquation(q=0.0)
print(f"same call without dividends: "
      f"{no_div.exact_solution(0.0, torch.tensor([[100.0]])).item():.4f}")
assert result["relative_error"] < 5.0, "did not converge — please open an issue!"

# %%
delta = solver.delta0().item()
exact_delta = equation.exact_gradient(0.0, torch.tensor([[100.0]])).item()
print(f"solver delta: {delta:.4f}   Merton delta: {exact_delta:.4f}")
assert abs(delta - exact_delta) < 0.05, "delta off — please open an issue!"

# %% [markdown]
# ## Where to go next
#
# - Sanity-check the sign conventions of *your* equation the way the
#   library does its own: a finite-difference PDE-residual test against a
#   known solution (`tests/test_convergence.py` shows the pattern — it
#   caught two real driver bugs before release).
# - Nonlinear drivers (the interesting ones) work the same way — see
#   `deep_fbsde_nn/equations/xva.py` for a production example with
#   $V^+$/$V^-$ terms and an in-class closed form.
# - Have an equation others would use? The repo seeds
#   [add-an-equation issues](https://github.com/ionutnodis/deep-fbsde-nn/issues) —
#   contributions welcome (`CONTRIBUTING.md`).
