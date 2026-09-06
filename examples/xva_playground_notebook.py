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
# # XVA playground — pricing with default risk and funding costs
#
# Price a European option the way a bank desk has to: the risk-free
# Black-Scholes value is only the starting point, and counterparty default
# risk (CVA), own default (DVA), and funding costs (FVA) adjust it. This
# notebook prices with the Burgard-Kjaer XVA model, checks the answer
# against an exact closed form, and reads off a hedge-quality delta —
# in a couple of minutes on a free Colab CPU.
#
# Generated from
# [`examples/xva_playground_notebook.py`](https://github.com/ionutnodis/deep-fbsde-nn/blob/main/examples/xva_playground_notebook.py)
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
import os

import torch

from deep_fbsde_nn.equations import XVAEquation
from deep_fbsde_nn.solvers import SolverConfig, StepwiseSolver

# CI executes this notebook with NB_FAST=1 (fewer iterations, one spot).
# The default settings below are the benchmark-grade run.
FAST = os.environ.get("NB_FAST") == "1"
ITERS, FINE_ITERS = (400, 200) if FAST else (2000, 1000)
torch.manual_seed(0)

# %% [markdown]
# ## The model, in four lines
#
# Burgard-Kjaer (2011) prices a derivative $V$ under bilateral default risk
# and a funding spread. The BSDE driver picks up the adjustment
#
# $$f_{\mathrm{xva}}(V) = \underbrace{\lambda_c (1-R_c)\, V^+}_{\text{CVA}}
# \;-\; \underbrace{\lambda_b (1-R_b)\, V^-}_{\text{DVA}}
# \;+\; \underbrace{(r_f - r)\, V}_{\text{FVA}}$$
#
# For a **long option** $V \ge 0$ on every path, so the nonlinear driver
# linearizes and the BSDE solves in closed form:
# $V = e^{-c(T-t)} \cdot \mathrm{BS}$ with cost rate
# $c = \lambda_c(1-R_c) + (r_f - r)$. That gives us an exact answer to
# grade the solver against.

# %%
equation = XVAEquation(dimension=1)  # Burgard-Kjaer defaults, ATM call, S0=K=100
print(equation)
print(f"effective cost rate c = {equation.effective_cost_rate:.4f} per year")

S0 = torch.tensor([[100.0]])
# Switch every adjustment off (no default risk, funding at the risk-free
# rate) and the model reduces exactly to Black-Scholes:
riskfree = XVAEquation(dimension=1, lambda_c=0.0, lambda_b=0.0, r_f=0.05)
bs_price = riskfree.exact_solution(0.0, S0).item()
xva_price_exact = equation.exact_solution(0.0, S0).item()
print(f"risk-free Black-Scholes price: {bs_price:.4f}")
print(f"exact XVA-adjusted price:      {xva_price_exact:.4f}")
print(f"total XVA adjustment:          {bs_price - xva_price_exact:+.4f}")

# %% [markdown]
# ## Solve the nonlinear BSDE
#
# The solver does **not** know about the linearization — it trains on the
# full nonlinear driver ($V^+$, $V^-$ and all) and has to find the closed
# form on its own. `StepwiseSolver` is the original Han-Jentzen-E
# parameterization: trainable $(Y_0, Z_0)$ plus one network per timestep.


# %%
def train_anchored(spot: float, iters: int = ITERS, fine_iters: int = FINE_ITERS):
    """Train a StepwiseSolver anchored at one spot. Returns (price, delta)."""
    # NOTE: pass the strike explicitly — it defaults to S0 * dimension (ATM),
    # which is a different contract at every spot.
    eq = XVAEquation(dimension=1, S0=spot, strike=100.0)
    config = SolverConfig(
        batch_size=64, num_timesteps=20, learning_rate=5e-3,
        num_iterations=iters, use_mlmc=False, print_every=10**9,
    )
    solver = StepwiseSolver(eq, config, device="cpu")
    solver.train()
    if fine_iters:  # fine-tune at a lower learning rate
        for group in solver.optimizer.param_groups:
            group["lr"] = 1e-3
        solver.train(n_iter=fine_iters)
    return eq, solver.predict().item(), solver.delta0().item()


equation, price, delta = train_anchored(100.0)
exact_price = equation.exact_solution(0.0, S0).item()
exact_delta = equation.exact_gradient(0.0, S0).item()

print(f"solver price {price:.4f}  vs exact {exact_price:.4f}  "
      f"({abs(price - exact_price) / exact_price * 100:.2f}% off)")
print(f"solver delta {delta:.4f}  vs exact {exact_delta:.4f}  "
      f"({abs(delta - exact_delta):.4f} abs off)")
assert abs(price - exact_price) / exact_price < 0.05, "did not converge — please open an issue!"
assert abs(delta - exact_delta) < 0.05, "delta off — please open an issue!"

# %% [markdown]
# ## Greeks that hold far from the money
#
# `StepwiseSolver` learns the solution *at its anchor point* — so the recipe
# for a hedge curve is to anchor one solver at each spot you care about.
# Each anchored run is its own well-scaled problem, which is why deep
# in/out-of-the-money deltas come out as accurate as at-the-money ones
# (validated by the slow test suite across S = 75-130: worst delta error
# ~0.01).

# %%
spots = [100.0] if FAST else [80.0, 100.0, 120.0]
rows = []
for spot in spots:
    eq, p, d = train_anchored(spot)
    x = torch.tensor([[spot]])
    rows.append((spot, p, eq.exact_solution(0.0, x).item(),
                 d, eq.exact_gradient(0.0, x).item()))

print(f"{'spot':>6} {'price':>9} {'exact':>9} {'delta':>8} {'exact':>8}")
for spot, p, pe, d, de in rows:
    print(f"{spot:>6.0f} {p:>9.4f} {pe:>9.4f} {d:>8.4f} {de:>8.4f}")
    assert abs(d - de) < 0.05, f"delta off at S={spot} — please open an issue!"

# %% [markdown]
# ## Turn the knobs
#
# The closed form makes XVA intuition instant — no retraining needed.
# Edit the scenarios (or add your own) and re-run: a riskier counterparty
# or a wider funding spread eats directly into the option's value.

# %%
scenarios = {
    "base case":               dict(),
    "risky counterparty":      dict(lambda_c=0.10),
    "funding squeeze":         dict(r_f=0.09),
    "no recovery, wide spread": dict(R_c=0.0, r_f=0.08),
}
print(f"{'scenario':<26} {'c (cost rate)':>13} {'price':>8} {'vs base':>8}")
base = XVAEquation(dimension=1).exact_solution(0.0, S0).item()
for label, overrides in scenarios.items():
    eq = XVAEquation(dimension=1, **overrides)
    v = eq.exact_solution(0.0, S0).item()
    print(f"{label:<26} {eq.effective_cost_rate:>13.4f} {v:>8.4f} {v - base:>+8.4f}")

# %% [markdown]
# ## Where to go next
#
# - Retrain the solver on your favorite scenario above and confirm it still
#   matches the closed form — the driver stays nonlinear, the check stays
#   honest.
# - At `dimension > 1` the payoff is on a basket and there is no closed
#   form (`exact_solution` returns `None`). The D=1 model itself is
#   cross-validated against a classical Monte-Carlo oracle in
#   `tests/test_xva.py`.
# - The quickstart notebook covers the solver basics; the custom-equation
#   notebook shows how to add your own model in ~30 lines.
