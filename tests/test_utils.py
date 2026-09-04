"""Utils: device selection, metrics, MC benchmarks, packaging metadata."""

import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

import deep_fbsde_nn
from deep_fbsde_nn.utils import get_device
from deep_fbsde_nn.utils.metrics import (
    ErrorTracker,
    max_absolute_error,
    mean_squared_error,
    monte_carlo_bsb_solution,
    monte_carlo_price,
    relative_error,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestDevice:
    def test_explicit_cpu(self):
        assert get_device("cpu") == torch.device("cpu")

    def test_auto_returns_a_device(self):
        assert isinstance(get_device("auto"), torch.device)

    def test_setup_device_is_read_only_by_default(self):
        from deep_fbsde_nn.utils.device import setup_device

        threads_before = torch.get_num_threads()
        caps = setup_device(torch.device("cpu"))
        assert torch.get_num_threads() == threads_before, "must not mutate by default"
        assert caps["device"].type == "cpu"

    def test_setup_device_optimize_opt_in(self):
        from deep_fbsde_nn.utils.device import setup_device

        threads_before = torch.get_num_threads()
        try:
            setup_device(torch.device("cpu"), optimize=True)
            assert torch.get_num_threads() == 8
        finally:
            torch.set_num_threads(threads_before)


class TestCLI:
    def test_cite_prints_bibtex_with_doi(self):
        result = subprocess.run(
            [sys.executable, "-m", "deep_fbsde_nn", "--cite"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "@software" in result.stdout
        assert "10.5281/zenodo.22311423" in result.stdout

    def test_bare_invocation_prints_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "deep_fbsde_nn"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "deep-fbsde-nn" in result.stdout


class TestMetrics:
    def test_relative_error_percentage(self):
        assert relative_error(110.0, 100.0) == pytest.approx(10.0)

    def test_relative_error_near_zero_reference(self):
        # Falls back to absolute error x100 when the reference is ~0
        assert relative_error(0.01, 0.0) == pytest.approx(1.0)

    def test_mse_and_max_abs(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 5.0])
        assert mean_squared_error(a, b) == pytest.approx(4.0 / 3.0)
        assert max_absolute_error(a, b) == pytest.approx(2.0)

    def test_error_tracker(self):
        tracker = ErrorTracker(exact_value=1.0)
        tracker.update(0, loss=0.5, prediction=1.1)
        tracker.update(1, loss=0.2, prediction=1.05)
        errors = tracker.get_relative_errors()
        assert errors[-1] == pytest.approx(5.0)
        assert tracker.final_error() == pytest.approx(5.0)
        assert tracker.to_dict()["iterations"] == [0, 1]


class TestMonteCarlo:
    def test_basket_call_with_tiny_vol_approaches_intrinsic(self):
        # sigma -> 0, r = 0: price -> max(mean(S0) - K, 0)
        price, stderr = monte_carlo_price(
            S0=1.2, K=1.0, T=1.0, r=0.0, sigma=1e-4, D=3, n_paths=10_000
        )
        assert abs(price - 0.2) < 1e-3
        assert stderr < 1e-3

    def test_unknown_payoff_raises(self):
        with pytest.raises(ValueError, match="Unknown payoff"):
            monte_carlo_price(1.0, 1.0, 1.0, 0.0, 0.2, 2, payoff_type="banana")

    def test_bsb_mc_matches_analytic(self):
        # E[||X_T||^2 | X_t] = ||x||^2 * exp(sigma^2 * tau) for this dynamics
        x = torch.ones(1, 2) / np.sqrt(2.0)  # ||x||^2 = 1
        sigma_max = 0.3
        mean_value, stderr = monte_carlo_bsb_solution(
            x, t=0.0, T=1.0, sigma_max=sigma_max, n_paths=200_000
        )
        exact = float(np.exp(sigma_max**2))
        assert abs(mean_value - exact) < max(5 * stderr, 0.02)


class TestPackagingMetadata:
    def test_version_single_source(self):
        if deep_fbsde_nn.__version__ == "0.0.0+unknown":
            pytest.skip("package not installed (bare source checkout)")
        with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
            pyproject = tomllib.load(fh)
        assert deep_fbsde_nn.__version__ == pyproject["project"]["version"]

    def test_experimental_package_warns_on_import(self):
        # -W error::UserWarning turns the quarantine warning into an error:
        # a nonzero exit proves the warning fires for anyone importing it.
        result = subprocess.run(
            [sys.executable, "-W", "error::UserWarning", "-c", "import experiments.experimental"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "experimental" in result.stderr
