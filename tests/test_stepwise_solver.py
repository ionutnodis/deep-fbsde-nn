"""StepwiseSolver: construction guards, training mechanics, persistence."""

import numpy as np
import pytest
import torch

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation, BlackScholesEquation
from deep_fbsde_nn.equations.hjb import HJBEquation
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StepwiseSolver

D = 3


def _config(**overrides):
    defaults = dict(
        batch_size=8, num_timesteps=6, learning_rate=1e-3,
        num_iterations=3, use_mlmc=False, print_every=10_000,
    )
    defaults.update(overrides)
    return SolverConfig(**defaults)


def _solver(**kwargs):
    eq = HJBEquation(dimension=D)
    return StepwiseSolver(eq, _config(), device="cpu", hidden_dim=8, **kwargs)


class TestConstructionGuards:
    def test_mlmc_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="use_mlmc=False"):
            StepwiseSolver(HJBEquation(dimension=D), _config(use_mlmc=True), device="cpu")

    def test_correlated_equations_are_rejected(self):
        eq = BlackScholesEquation(dimension=D, correlation=0.5)
        with pytest.raises(NotImplementedError, match="correlated"):
            StepwiseSolver(eq, _config(), device="cpu")

    def test_one_z_net_per_interior_step(self):
        solver = _solver()
        assert len(solver.model.z_nets) == solver.config.num_timesteps - 1
        assert solver.model.Y0.shape == (1, 1)
        assert solver.model.Z0.shape == (1, D)

    def test_naisnet_subnets_supported(self):
        solver = _solver(network_cls=NAISNet)
        history = solver.train(n_iter=2)
        assert np.isfinite(history["losses"]).all()


class TestTraining:
    def test_training_updates_y0_and_is_finite(self):
        solver = _solver()
        y0_before = solver.model.Y0.item()
        history = solver.train(n_iter=3)
        assert np.isfinite(history["losses"]).all()
        assert solver.model.Y0.item() != y0_before

    def test_chunked_resume(self):
        solver = _solver()
        solver.train(n_iter=2)
        opt = solver.optimizer
        solver.train(n_iter=2)
        assert solver.current_iteration == 4
        assert solver.optimizer is opt

    def test_validate_reports_error_against_reference(self):
        solver = _solver()
        solver.train(n_iter=2)
        result = solver.validate()
        assert set(result) >= {"Y0_pred", "Y0_exact", "relative_error"}

    def test_validate_without_reference(self):
        eq = BlackScholesEquation(dimension=D)  # basket: MC-method only
        solver = StepwiseSolver(eq, _config(), device="cpu", hidden_dim=8)
        result = solver.validate()
        assert "Y0_pred" in result and "Y0_exact" not in result


class TestPredictContract:
    def test_predict_returns_y0_at_origin(self):
        solver = _solver()
        assert solver.predict().shape == (1, 1)
        assert solver.predict(0.0, solver.X0).item() == solver.model.Y0.item()

    def test_predict_rejects_nonzero_time(self):
        with pytest.raises(ValueError, match="t=0"):
            _solver().predict(0.5)

    def test_predict_rejects_other_points(self):
        solver = _solver()
        with pytest.raises(ValueError, match="fixed X0"):
            solver.predict(0.0, solver.X0 + 1.0)


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        solver = _solver()
        solver.train(n_iter=3)
        path = str(tmp_path / "stepwise.pt")
        solver.save(path)

        fresh = _solver()
        fresh.load(path)
        assert fresh.model.Y0.item() == solver.model.Y0.item()
        for k, v in solver.model.state_dict().items():
            assert torch.equal(v, fresh.model.state_dict()[k])
        assert fresh.current_iteration == 3


def test_bsb_smoke():
    eq = BlackScholesBarenblattEquation(dimension=2)
    solver = StepwiseSolver(eq, _config(), device="cpu", hidden_dim=8)
    history = solver.train(n_iter=3)
    assert np.isfinite(history["losses"]).all()
