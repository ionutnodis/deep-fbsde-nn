"""GlobalSolver: initial-condition sampling strategies and training smoke."""

import numpy as np
import pytest
import torch

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import GlobalSolver, SolverConfig

D = 2
CENTER = 1.0
SPREAD = 0.1


def _solver(sampling):
    eq = BlackScholesBarenblattEquation(dimension=D)
    net = NAISNet(input_dim=D + 1, hidden_dim=16, output_dim=1, num_layers=3)
    config = SolverConfig(
        batch_size=4, num_timesteps=5, num_iterations=2, use_mlmc=False, print_every=10_000
    )
    return GlobalSolver(
        eq, net, config, device="cpu", sampling=sampling, X0_center=CENTER, X0_spread=SPREAD
    )


@pytest.mark.parametrize("sampling", ["fixed", "uniform", "lognormal", "gaussian"])
def test_sampling_shapes(sampling):
    solver = _solver(sampling)
    X0 = solver._get_initial_condition(500)
    assert X0.shape == (500, D)
    assert torch.isfinite(X0).all()


def test_fixed_sampling_returns_center():
    X0 = _solver("fixed")._get_initial_condition(50)
    assert torch.allclose(X0, torch.full((50, D), CENTER))


def test_uniform_sampling_respects_bounds():
    X0 = _solver("uniform")._get_initial_condition(2000)
    assert X0.min().item() >= CENTER * (1 - SPREAD) - 1e-6
    assert X0.max().item() <= CENTER * (1 + SPREAD) + 1e-6


def test_lognormal_sampling_is_positive():
    X0 = _solver("lognormal")._get_initial_condition(2000)
    assert (X0 > 0).all()


def test_gaussian_sampling_centers_correctly():
    X0 = _solver("gaussian")._get_initial_condition(20_000)
    # standard error = spread * center / sqrt(n)
    assert abs(X0.mean().item() - CENTER) < 5 * SPREAD * CENTER / np.sqrt(20_000 * D)


def test_unknown_sampling_raises():
    solver = _solver("fixed")
    solver.sampling = "banana"
    with pytest.raises(ValueError, match="Unknown sampling"):
        solver._get_initial_condition(4)


def test_training_smoke():
    solver = _solver("lognormal")
    history = solver.train(n_iter=2)
    assert np.isfinite(history["losses"]).all()
