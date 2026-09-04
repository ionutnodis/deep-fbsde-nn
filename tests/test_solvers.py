"""Solver mechanics: MLMC schedule, path statistics, persistence, resume."""

import numpy as np
import pytest
import torch

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation, VanillaCallEquation
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StandardSolver


def _small_solver(dimension=2, **config_overrides):
    defaults = dict(
        batch_size=4,
        num_timesteps=6,
        learning_rate=1e-3,
        num_iterations=3,
        use_mlmc=False,
        print_every=10_000,
    )
    defaults.update(config_overrides)
    eq = BlackScholesBarenblattEquation(dimension=dimension)
    net = NAISNet(input_dim=dimension + 1, hidden_dim=16, output_dim=1, num_layers=3)
    return StandardSolver(eq, net, SolverConfig(**defaults), device="cpu")


class TestMLMCSchedule:
    def test_disabled_returns_configured_timesteps(self):
        solver = _small_solver(use_mlmc=False, num_timesteps=42)
        assert solver._get_mlmc_timesteps(0) == 42
        assert solver._get_mlmc_timesteps(99_999) == 42

    def test_schedule_thresholds(self):
        solver = _small_solver(use_mlmc=True, mlmc_schedule={0: 5, 10: 7, 20: 9})
        assert solver._get_mlmc_timesteps(0) == 5
        assert solver._get_mlmc_timesteps(9) == 5
        assert solver._get_mlmc_timesteps(10) == 7
        assert solver._get_mlmc_timesteps(19) == 7
        assert solver._get_mlmc_timesteps(20) == 9
        assert solver._get_mlmc_timesteps(10_000) == 9


class TestGeneratePaths:
    def test_shapes_and_grid(self):
        solver = _small_solver(dimension=3)
        M, N = 8, 5
        t, dW, W = solver.generate_paths(M, N)
        T = solver.equation.T
        assert t.shape == (N + 1,)
        assert t[0].item() == 0.0
        assert abs(t[-1].item() - T) < 1e-6
        assert dW.shape == (M, N, 3)
        assert W.shape == (M, N + 1, 3)
        assert torch.all(W[:, 0, :] == 0)
        assert torch.allclose(W[:, 1:, :], torch.cumsum(dW, dim=1), atol=1e-6)

    def test_brownian_statistics(self):
        solver = _small_solver(dimension=2)
        M, N = 20_000, 4
        _, dW, _ = solver.generate_paths(M, N)
        dt = solver.equation.T / N
        # E[dW] = 0 within ~4 standard errors; Var[dW] = dt within 5%
        standard_error = np.sqrt(dt / M)
        assert dW.mean().abs().item() < 4 * standard_error
        assert abs(dW.var().item() - dt) / dt < 0.05


class TestPersistence:
    def test_save_load_round_trip_with_weights_only(self, tmp_path):
        solver = _small_solver()
        solver.train(n_iter=3)
        path = str(tmp_path / "ckpt.pt")
        solver.save(path)

        fresh = _small_solver()
        fresh.load(path)

        for k, v in solver.model.state_dict().items():
            assert torch.equal(v, fresh.model.state_dict()[k])
        assert fresh.current_iteration == solver.current_iteration
        assert fresh.training_history["iterations"] == solver.training_history["iterations"]

    def test_load_missing_file_raises(self, tmp_path):
        solver = _small_solver()
        with pytest.raises((FileNotFoundError, RuntimeError)):
            solver.load(str(tmp_path / "does-not-exist.pt"))


class TestTraining:
    def test_chunked_resume_preserves_state(self):
        solver = _small_solver()
        solver.train(n_iter=2)
        optimizer_first = solver.optimizer
        solver.train(n_iter=2)
        assert solver.current_iteration == 4
        assert solver.optimizer is optimizer_first, "optimizer must survive chunked training"

    def test_batch_greater_than_one_and_d1(self):
        eq = VanillaCallEquation(dimension=1)
        net = NAISNet(input_dim=2, hidden_dim=8, output_dim=1, num_layers=3)
        config = SolverConfig(
            batch_size=8, num_timesteps=4, num_iterations=2, use_mlmc=False, print_every=10_000
        )
        solver = StandardSolver(eq, net, config, device="cpu")
        history = solver.train(n_iter=2)
        assert np.isfinite(history["losses"]).all()

    def test_predict_unsqueezes_1d_input(self):
        solver = _small_solver(dimension=2)
        out = solver.predict(0.0, torch.ones(2))
        assert out.shape == (1, 1)

    def test_validate_reports_error_when_exact_exists(self):
        solver = _small_solver()
        result = solver.validate()
        assert "Y0_pred" in result
        assert "Y0_exact" in result
        assert "relative_error" in result
        assert np.isfinite(result["Y0_exact"])

    def test_repr_is_informative(self):
        solver = _small_solver()
        text = repr(solver)
        assert "StandardSolver" in text
        assert "params=" in text
