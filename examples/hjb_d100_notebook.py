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
# # Reproduce a published result: the d=100 HJB value of Han, Jentzen & E
#
# The paper that started the Deep BSDE field — Han, Jentzen & E,
# *"Solving high-dimensional partial differential equations using deep
# learning"* (PNAS 2018) — reports $u(0, \mathbf{0}) = 4.5901$ for a
# 100-dimensional Hamilton-Jacobi-Bellman equation. This notebook
# reproduces that number on a free Colab CPU (a few minutes), and grades
# it against an independent Monte-Carlo reference you compute yourself.
#
# Generated from
# [`examples/hjb_d100_notebook.py`](https://github.com/ionutnodis/deep-fbsde-nn/blob/main/examples/hjb_d100_notebook.py)
# (jupytext percent format) and **executed in CI** (at the `NB_FAST=1`
# setting, d=20), so it cannot silently rot.

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

from deep_fbsde_nn.equations import HJBEquation
from deep_fbsde_nn.solvers import SolverConfig, StepwiseSolver

# CI executes this notebook with NB_FAST=1: same code path, d=20, fewer
# iterations (~30s). The default below is the full published-value run.
FAST = os.environ.get("NB_FAST") == "1"
DIM = 20 if FAST else 100
ITERS = 1500 if FAST else 3000
N_MC = 200_000 if FAST else 400_000
torch.manual_seed(0)

# %% [markdown]
# ## The equation, and a reference you can check independently
#
# The HJB equation from the paper (a stochastic control problem):
#
# $$\partial_t u + \Delta u - \lambda \|\nabla u\|^2 = 0, \qquad
# u(T, x) = g(x) = \ln\!\big(\tfrac{1}{2}(1 + \|x\|^2)\big)$$
#
# What makes it a great benchmark: the Cole-Hopf transform turns it into a
# *linear* problem, giving the exact solution as an expectation
#
# $$u(t, x) = -\tfrac{1}{\lambda}
# \ln \mathbb{E}\big[\exp(-\lambda\, g(x + \sqrt{2}\, W_{T-t}))\big]$$
#
# which plain Monte-Carlo can evaluate to high accuracy — no neural network
# involved. So the reference below is independent of everything the solver
# does.

# %%
equation = HJBEquation(dimension=DIM)
X0 = equation.sample_initial_condition(1)  # the origin
reference = equation.exact_solution(
    0.0, X0, n_mc=N_MC, generator=torch.Generator().manual_seed(0)
).item()
print(f"Cole-Hopf Monte-Carlo reference (d={DIM}, {N_MC:,} paths): {reference:.4f}")
if DIM == 100:
    print("published value (Han, Jentzen & E 2018):                  4.5901")

# %% [markdown]
# ## Train the solver
#
# `StepwiseSolver` is the paper's own parameterization: a trainable
# $(Y_0, Z_0)$ plus one small network per timestep. The quadratic driver
# $-\lambda\|Z\|^2$ punishes noisy gradients hard at high dimension, so the
# recipe is a **larger batch and smaller learning rate** than you'd use at
# d=3 (this exact configuration is the library's benchmark row: 0.03% off
# the published value on a laptop CPU).

# %%
config = SolverConfig(
    batch_size=256,
    num_timesteps=20,
    learning_rate=5e-4,
    num_iterations=ITERS,
    use_mlmc=False,
    print_every=500,
)
solver = StepwiseSolver(equation, config, device="cpu", hidden_dim=DIM + 10)
solver.train()

# %% [markdown]
# ## The verdict

# %%
prediction = solver.predict().item()
rel_error = abs(prediction - reference) / abs(reference) * 100
print(f"solver u(0, 0):     {prediction:.4f}")
print(f"MC reference:       {reference:.4f}")
print(f"relative error:     {rel_error:.2f}%")
if DIM == 100:
    published = 4.5901
    print(f"published value:    {published:.4f} "
          f"({abs(prediction - published) / published * 100:.2f}% off)")
assert rel_error < 3.0, "did not converge — please open an issue!"

# %% [markdown]
# ## Where to go next
#
# - Drop `DIM` to 3 and watch the same code converge in seconds — or push
#   past 100 (the paper's point: no grid, so dimension is a parameter, not
#   a wall).
# - The library's driver convention was validated with a finite-difference
#   PDE-residual test against this same Cole-Hopf reference — the test that
#   caught two historical sign/scaling bugs (`tests/test_convergence.py`).
# - Reproduce the full 12-row benchmark table with one command:
#   `python benchmarks/run.py` in a clone.
