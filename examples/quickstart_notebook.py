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
# # deep-fbsde-nn quickstart
#
# Solve a high-dimensional PDE with the Deep BSDE method and check the answer
# against the exact solution — in about a minute on a free Colab CPU.
#
# This notebook is generated from
# [`examples/quickstart_notebook.py`](https://github.com/ionutnodis/deep-fbsde-nn/blob/main/examples/quickstart_notebook.py)
# (jupytext percent format) and **executed in CI on every push**, so it cannot
# silently rot.

# %%
try:
    import deep_fbsde_nn
except ImportError:  # e.g. on Colab
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "deep-fbsde-nn"], check=True)
    import deep_fbsde_nn

print("deep-fbsde-nn", deep_fbsde_nn.__version__)

# %% [markdown]
# ## The problem
#
# The Black-Scholes-Barenblatt equation (option pricing under uncertain
# volatility) has a known exact solution
# $u(t,x) = \|x\|^2 e^{\sigma_{\max}^2 (T-t)}$ — ideal for validating the
# solver. We use a small dimension here so the notebook runs fast; the same
# code at `dimension=100` reproduces the benchmark table in the README.

# %%
import torch

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation
from deep_fbsde_nn.solvers import SolverConfig, StepwiseSolver

torch.manual_seed(0)

dimension = 10
equation = BlackScholesBarenblattEquation(dimension=dimension)

config = SolverConfig(
    batch_size=128,
    num_timesteps=20,
    learning_rate=1e-3,
    num_iterations=800,   # bump to ~2000 for benchmark-grade accuracy
    use_mlmc=False,
    print_every=200,
)

solver = StepwiseSolver(equation, config, device="cpu")
solver.train()

# %% [markdown]
# ## Validate against the exact solution

# %%
result = solver.validate()
print(f"predicted u(0, x0): {result['Y0_pred']:.5f}")
print(f"exact u(0, x0):     {result['Y0_exact']:.5f}")
print(f"relative error:     {result['relative_error']:.2f}%")
assert result["relative_error"] < 5.0, "did not converge — please open an issue!"

# %% [markdown]
# ## Where to go next
#
# - `StandardSolver` / `GlobalSolver` learn the full surface $u(t,x)$
#   (needed for greeks) — see the README's solver table for the trade-offs.
# - Five equations ship validated against references:
#   Black-Scholes(-Barenblatt), vanilla call, Allen-Cahn, HJB.
# - Reproduce the full benchmark: `python benchmarks/run.py` in a clone.
# - Write your own equation in ~20 lines: README → "Extending the Library".
