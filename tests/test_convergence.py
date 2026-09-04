"""Convergence against known solutions. Marked slow — CI runs these on tag + weekly.

Tolerances are correctness-level (not run-reproduction): loose enough to
survive torch minor upgrades, tight enough that a broken solver fails.
Budgets are calibrated for CPU CI runners.

Division of labor for the HJB equation:
- ``test_hjb_driver_consistent_with_reference`` is the sharp discriminator.
  It finite-differences the Cole-Hopf MC reference and checks the PDE
  residual using the *implemented* driver. The two real bugs this suite has
  caught (a driver sign flip in StandardSolver and a factor-2 error in the
  HJB driver) produce residuals 13x-40x above the threshold.
- ``test_hjb_end_to_end_sanity`` trains the full pipeline but asserts only a
  loose band: with Z derived from the same network as u (autograd), the
  optimizer damps the martingale term by under-estimating ||Z||, which
  under-weights the quadratic driver — a known limitation of this
  architecture on strongly nonlinear drivers (Han et al. used per-timestep
  Z subnetworks). Tightening this band is a tracked TODO (per-timestep Z).
"""

import numpy as np
import pytest
import torch

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation, VanillaCallEquation
from deep_fbsde_nn.equations.hjb import HJBEquation
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StandardSolver

pytestmark = pytest.mark.slow


def _train(equation, hidden=32, layers=3, iters=1500, fine_iters=700, lr=5e-3,
           fine_lr=1e-3, batch=16, timesteps=16):
    torch.manual_seed(0)
    np.random.seed(0)
    net = NAISNet(
        input_dim=equation.D + 1, hidden_dim=hidden, output_dim=1, num_layers=layers
    )
    config = SolverConfig(
        batch_size=batch,
        num_timesteps=timesteps,
        learning_rate=lr,
        num_iterations=iters,
        use_mlmc=False,
        print_every=1000,
    )
    solver = StandardSolver(equation, net, config, device="cpu")
    solver.train(n_iter=iters)
    for group in solver.optimizer.param_groups:
        group["lr"] = fine_lr
    solver.train(n_iter=fine_iters)
    return solver


def test_bsb_low_dimension_converges():
    eq = BlackScholesBarenblattEquation(dimension=2)
    solver = _train(eq, iters=600, fine_iters=0)
    result = solver.validate()
    # Exact: ||x0||^2 * exp(sigma_max^2 * T) = exp(0.09) = 1.09417
    assert result["relative_error"] < 2.0, result


def test_vanilla_call_d1_converges_to_black_scholes():
    eq = VanillaCallEquation(dimension=1)
    solver = _train(eq, iters=1500, fine_iters=700)
    result = solver.validate()
    # Exact: 0.104035 (calibrated run lands at ~1% error)
    assert result["relative_error"] < 5.0, result


def test_hjb_driver_consistent_with_reference():
    """Finite-difference the MC reference; the residual of
    u_t + Lap(u) + driver(Z=grad u) must vanish.

    Measured residual with the correct driver: ~4e-4.
    Sign-flipped driver: ~6.6e-2. Factor-2 driver: ~1.7e-2.
    """
    eq = HJBEquation(dimension=3)
    x0 = torch.tensor([[0.5, 0.5, 0.5]])
    n_mc = 800_000

    def u(t, x, seed=11):
        return eq.exact_solution(
            t, x, n_mc=n_mc, generator=torch.Generator().manual_seed(seed)
        ).item()

    dt, eps = 0.01, 0.02
    u_0 = u(0.0, x0)
    du_dt = (u(dt, x0) - u_0) / dt

    laplacian = 0.0
    grad = torch.zeros(1, eq.D)
    for i in range(eq.D):
        e = torch.zeros(1, eq.D)
        e[0, i] = eps
        up, dn = u(0.0, x0 + e), u(0.0, x0 - e)
        laplacian += (up - 2 * u_0 + dn) / eps**2
        grad[0, i] = (up - dn) / (2 * eps)

    f = eq.driver(torch.tensor(0.0), x0, torch.tensor([[u_0]]), grad).item()
    residual = du_dt + laplacian + f
    assert abs(residual) < 5e-3, {
        "residual": residual, "du_dt": du_dt, "laplacian": laplacian, "driver": f
    }


def test_hjb_end_to_end_sanity():
    eq = HJBEquation(dimension=3)
    solver = _train(eq, hidden=64, batch=64, timesteps=20, iters=2000, fine_iters=1000)
    y0_pred = solver.predict(0.0, eq.sample_initial_condition(1)).item()
    reference = eq.exact_solution(
        0.0, eq.sample_initial_condition(1), n_mc=200_000,
        generator=torch.Generator().manual_seed(0),
    ).item()
    rel_error = abs(y0_pred - reference) / abs(reference) * 100
    # Loose sanity band (see module docstring): calibrated runs land ~12-15%;
    # the historical sign bug produced ~36%. Tightening tracked in TODOS.md.
    assert rel_error < 20.0, (y0_pred, reference, rel_error)
