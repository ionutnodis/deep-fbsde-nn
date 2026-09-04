"""Equation contracts: shapes, terminal conditions, and exact solutions."""

import pytest
import torch

from deep_fbsde_nn.equations import (
    AllenCahnEquation,
    BlackScholesBarenblattEquation,
    BlackScholesEquation,
    HJBEquation,
    VanillaCallEquation,
)

M = 4  # batch size for shape checks
D = 3


def _make(equation_cls):
    return equation_cls(dimension=D)


ALL_EQUATIONS = [
    BlackScholesEquation,
    BlackScholesBarenblattEquation,
    AllenCahnEquation,
    VanillaCallEquation,
    HJBEquation,
]


@pytest.mark.parametrize("equation_cls", ALL_EQUATIONS)
def test_shapes(equation_cls, device):
    eq = _make(equation_cls)
    t = torch.tensor(0.0)
    X = eq.sample_initial_condition(M) + 0.1  # off-center, still valid
    Y = torch.rand(M, 1)
    Z = torch.rand(M, D)

    assert eq.drift(t, X, Y, Z).shape == (M, D)
    assert eq.diffusion(t, X, Y).shape == (M, D)
    assert eq.driver(t, X, Y, Z).shape == (M, 1)
    assert eq.terminal(X).shape == (M, 1)
    assert eq.sample_initial_condition(M).shape == (M, D)


class TestBlackScholesBarenblatt:
    def test_exact_solution_at_terminal_equals_payoff(self):
        eq = _make(BlackScholesBarenblattEquation)
        X = torch.rand(M, D) + 0.5
        exact = eq.exact_solution(eq.T, X)
        assert torch.allclose(exact, eq.terminal(X), atol=1e-6)

    def test_exact_solution_at_zero(self):
        eq = BlackScholesBarenblattEquation(dimension=D, sigma_max=0.3)
        X = eq.sample_initial_condition(1)  # ||x||^2 == 1
        expected = torch.exp(torch.tensor(0.3**2 * eq.T))
        assert torch.allclose(exact := eq.exact_solution(0.0, X), expected.view(1, 1), atol=1e-5)
        assert exact.shape == (1, 1)

    def test_exact_gradient_shape_and_value(self):
        eq = _make(BlackScholesBarenblattEquation)
        X = torch.rand(M, D)
        grad = eq.exact_gradient(eq.T, X)
        assert grad.shape == (M, D)
        assert torch.allclose(grad, 2.0 * X, atol=1e-6)

    def test_has_exact_solution(self):
        assert _make(BlackScholesBarenblattEquation).has_exact_solution()


class TestVanillaCall:
    """The closed form is exact only at D=1 (the payoff is a basket sum)."""

    def test_atm_value_matches_black_scholes(self):
        # S=1, K=1, r=0.01, sigma=0.25, T=1 -> C = 0.104035 (textbook value)
        eq = VanillaCallEquation(dimension=1)
        value = eq.exact_solution(0.0, torch.tensor([[1.0]]))
        assert value.shape == (1, 1)
        assert abs(value.item() - 0.104035) < 5e-4

    def test_deep_itm_approaches_forward_value(self):
        # C -> S - K * exp(-r T) for S >> K
        eq = VanillaCallEquation(dimension=1)
        value = eq.exact_solution(0.0, torch.tensor([[3.0]])).item()
        assert abs(value - (3.0 - 0.99004983)) < 1e-3  # K * exp(-r*T) = 0.99005

    def test_deep_otm_is_nearly_zero(self):
        eq = VanillaCallEquation(dimension=1)
        assert eq.exact_solution(0.0, torch.tensor([[0.1]])).item() < 1e-6

    def test_terminal_time_returns_payoff(self):
        eq = VanillaCallEquation(dimension=1)
        value = eq.exact_solution(eq.T, torch.tensor([[1.3]]))
        assert torch.allclose(value, torch.tensor([[0.3]]), atol=1e-6)

    def test_multi_asset_has_no_closed_form(self):
        eq = VanillaCallEquation(dimension=2)
        assert eq.exact_solution(0.0, torch.ones(1, 2)) is None
        assert not eq.has_exact_solution()

    def test_single_asset_reports_exact_solution(self):
        assert VanillaCallEquation(dimension=1).has_exact_solution()


class TestHJB:
    def test_terminal_time_equals_payoff(self):
        # tau = 0 -> X_T == X deterministically -> u == g exactly
        eq = HJBEquation(dimension=D)
        X = torch.rand(1, D)
        value = eq.exact_solution(eq.T, X, n_mc=100)
        assert torch.allclose(value, eq.terminal(X), atol=1e-5)

    def test_seeded_reproducibility(self):
        eq = HJBEquation(dimension=D)
        X = torch.zeros(1, D)
        v1 = eq.exact_solution(0.0, X, n_mc=5000, generator=torch.Generator().manual_seed(7))
        v2 = eq.exact_solution(0.0, X, n_mc=5000, generator=torch.Generator().manual_seed(7))
        assert v1.item() == v2.item()

    def test_partial_chunk_is_unbiased(self):
        # n_mc=25000 exercises the final partial chunk (chunk size 10000).
        # Before the fix, the loop drew 30000 samples but divided by 25000,
        # biasing the expectation by x1.2. Compare against a multiple-of-chunk
        # run with the same seed budget; both estimate the same quantity.
        eq = HJBEquation(dimension=D)
        X = torch.zeros(1, D)
        v_partial = eq.exact_solution(
            0.0, X, n_mc=25000, generator=torch.Generator().manual_seed(3)
        ).item()
        v_full = eq.exact_solution(
            0.0, X, n_mc=40000, generator=torch.Generator().manual_seed(4)
        ).item()
        assert abs(v_partial - v_full) < 0.05  # ~MC noise, nowhere near a x1.2 bias

    def test_log_underflow_is_guarded(self):
        # Large lambda * positive g underflows exp(-lambda*g) to 0 for every
        # sample; the clamp keeps the log finite.
        eq = HJBEquation(dimension=D, lambda_val=1e8)
        X = torch.full((1, D), 10.0)
        value = eq.exact_solution(0.0, X, n_mc=200, generator=torch.Generator().manual_seed(0))
        assert torch.isfinite(value).all()

    def test_default_x0(self):
        eq = HJBEquation(dimension=D)
        value = eq.exact_solution(0.9, n_mc=1000, generator=torch.Generator().manual_seed(0))
        assert value.shape == (1, 1)
        assert torch.isfinite(value).all()


class TestAllenCahn:
    def test_terminal_at_origin(self):
        # canonical benchmark payoff: g(0) = 1/(2 + 0) = 0.5
        eq = AllenCahnEquation(dimension=D)
        assert torch.allclose(eq.terminal(torch.zeros(1, D)), torch.full((1, 1), 0.5))

    def test_terminal_gradient_matches_analytic(self):
        eq = AllenCahnEquation(dimension=D)
        X = torch.rand(4, D)
        norm_sq = (X**2).sum(dim=1, keepdim=True)
        expected = -0.8 * X / (2.0 + 0.4 * norm_sq) ** 2
        assert torch.allclose(eq.terminal_gradient(X), expected, atol=1e-6)

    def test_canonical_diffusion_is_sqrt2(self):
        eq = AllenCahnEquation(dimension=D)
        sigma = eq.diffusion(torch.tensor(0.0), torch.zeros(1, D), None)
        assert torch.allclose(sigma, torch.full((1, D), 2.0**0.5))

    def test_driver_follows_documented_convention(self):
        # PDE: du/dt + (1/2)Δu + (u - u³) = 0 -> driver returns f = u - u³
        # (dY = -f dt + Z^T sigma dW; same convention across all equations).
        eq = AllenCahnEquation(dimension=D)
        Y = torch.tensor([[0.5]])
        expected = 0.5 - 0.5**3
        assert torch.allclose(
            eq.driver(torch.tensor(0.0), torch.zeros(1, D), Y, torch.zeros(1, D)),
            torch.tensor([[expected]]),
        )
