"""XVA experiment validated against the classical Monte-Carlo oracle.

The classical reference in the same file was itself verified against closed
forms (CVA 0.124286 vs 0.124161, FVA 0.104611 vs 0.104506) during the v0.2
diagnosis. This test guards the two things the historical "Debugging
required" commit was about:
- the price (a driver-sign error once produced a ~0.46 gap), and
- the component breakdown (a missing consistency loss once left u(t>0, ·)
  as pure gauge, inventing DVA from nothing and flipping FVA's sign).

Requires the [experiments] extra (scipy, matplotlib) — skipped without it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp_xva import (  # noqa: E402
    XVABSDESolver,
    XVAParams,
    classical_xva_monte_carlo,
)

pytestmark = pytest.mark.slow


def test_xva_price_and_components_match_classical_mc():
    torch.manual_seed(0)
    np.random.seed(0)

    params = XVAParams()
    classical = classical_xva_monte_carlo(
        S0=params.S0, K=params.K, T=params.T, r=params.r, sigma=params.sigma,
        lambda_c=params.lambda_c, lambda_b=params.lambda_b,
        R_c=params.R_c, R_b=params.R_b, r_f=params.r_f,
        n_paths=200_000, n_steps=200,
    )

    solver = XVABSDESolver(params, hidden_dim=64, num_layers=2, device="cpu")
    for _ in range(15_000):
        solver.optimizer.zero_grad()
        loss, y0 = solver.compute_loss(batch_size=256, n_steps=10)
        loss.backward()
        solver.optimizer.step()
        solver.scheduler.step()
        solver.prices.append(y0)

    y0_final = float(np.mean(solver.prices[-500:]))
    rel_error = abs(y0_final - classical["XVA_price"]) / classical["XVA_price"] * 100
    # Calibrated run: 4.07%. The sign-bug regression this guards against
    # produced ~4.5x this (0.46 absolute on a 10.22 price).
    assert rel_error < 6.0, (y0_final, classical["XVA_price"], rel_error)

    components = solver.compute_xva_components(n_paths=50_000)
    # Calibrated: CVA 0.1272 vs 0.1243, FVA 0.1054 vs 0.1046, DVA 0.0004.
    # The gauge bug this guards against reported DVA +0.138 and FVA sign
    # flipped, while CVA went negative.
    assert abs(components["DVA"]) < 0.01, components
    assert abs(components["CVA"] - classical["CVA"]) / classical["CVA"] < 0.15, components
    assert abs(components["FVA"] - classical["FVA"]) / classical["FVA"] < 0.15, components
    assert components["CVA"] > 0 and components["FVA"] > 0, components


def test_xva_equation_anchored_price_and_delta():
    """The library-native XVA path (v0.3): anchor a StepwiseSolver at each
    query spot and read predict() / delta0().

    Every anchored run is its own well-scaled problem, so far-from-the-money
    points are as accurate as at-the-money — this closes the 0.2 caveat
    (delta at S=80 was +0.09 off in the bespoke experiment; anchored runs
    measure <= 0.011 across S=75..130). Full five-point calibration lives in
    the PR record; CI runs three points to stay inside the budget.
    """
    from deep_fbsde_nn.equations import XVAEquation
    from deep_fbsde_nn.solvers import SolverConfig, StepwiseSolver

    for spot in (75.0, 100.0, 130.0):
        torch.manual_seed(0)
        np.random.seed(0)
        eq = XVAEquation(dimension=1, S0=spot, strike=100.0)
        config = SolverConfig(
            batch_size=64, num_timesteps=20, learning_rate=5e-3,
            num_iterations=2000, use_mlmc=False, print_every=10**9,
        )
        solver = StepwiseSolver(eq, config, device="cpu", hidden_dim=32)
        solver.train(n_iter=2000)
        for group in solver.optimizer.param_groups:
            group["lr"] = 1e-3
        solver.train(n_iter=1000)

        X = torch.tensor([[spot]])
        price = solver.predict().item()
        delta = solver.delta0().item()
        exact_price = eq.exact_solution(0.0, X).item()
        exact_delta = eq.exact_gradient(0.0, X).item()

        # Calibrated: worst price 6.25% (deep OTM, 0.06 absolute), worst
        # delta error 0.011 across the full range.
        price_ok = (
            abs(price - exact_price) / exact_price < 0.08
            or abs(price - exact_price) < 0.1
        )
        assert price_ok, (spot, price, exact_price)
        assert abs(delta - exact_delta) < 0.02, (spot, delta, exact_delta)
