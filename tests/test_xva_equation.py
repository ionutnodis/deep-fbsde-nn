"""XVAEquation contracts: driver, closed form, gradients, guards."""

import pytest
import torch

from deep_fbsde_nn.equations import BlackScholesEquation, XVAEquation

S0 = torch.tensor([[100.0]])


class TestDriver:
    def test_reduces_to_black_scholes_without_adjustments(self):
        xva = XVAEquation(dimension=1, lambda_c=0.0, lambda_b=0.0, r_f=0.05, r=0.05)
        bs = BlackScholesEquation(dimension=1, r=0.05, sigma=0.2, strike=100.0)
        Y = torch.tensor([[7.3], [-2.1]])
        assert torch.allclose(
            xva.driver(0.0, S0.expand(2, 1), Y, None),
            bs.driver(0.0, S0.expand(2, 1), Y, None),
        )

    def test_xva_costs_components(self):
        eq = XVAEquation(dimension=1)
        V = torch.tensor([[10.0]])
        # V > 0: cva + fva, no dva
        expected = 0.02 * 0.6 * 10.0 + 0.01 * 10.0
        assert torch.allclose(eq.xva_costs(V), torch.tensor([[expected]]))
        # V < 0: -dva + fva
        Vn = torch.tensor([[-10.0]])
        expected_n = -0.01 * 0.6 * 10.0 + 0.01 * -10.0
        assert torch.allclose(eq.xva_costs(Vn), torch.tensor([[expected_n]]))

    def test_effective_cost_rate(self):
        eq = XVAEquation(dimension=1)
        assert eq.effective_cost_rate == pytest.approx(0.02 * 0.6 + 0.01)


class TestExactSolution:
    def test_matches_mc_cross_validated_value(self):
        # 10.2232 = e^{-0.022} * BS(10.4506); classical MC oracle gave 10.2217
        # (0.2 diagnosis) — agreement to MC error.
        eq = XVAEquation(dimension=1)
        assert eq.exact_solution(0.0, S0).item() == pytest.approx(10.2232, abs=2e-3)

    def test_costs_reduce_the_price(self):
        eq = XVAEquation(dimension=1)
        clean = XVAEquation(dimension=1, lambda_c=0.0, lambda_b=0.0, r_f=0.05)
        assert eq.exact_solution(0.0, S0).item() < clean.exact_solution(0.0, S0).item()

    def test_terminal_time_returns_payoff(self):
        eq = XVAEquation(dimension=1)
        S = torch.tensor([[130.0]])
        assert torch.allclose(eq.exact_solution(eq.T, S), torch.tensor([[30.0]]), atol=1e-4)

    def test_put_value_is_positive_and_sane(self):
        eq = XVAEquation(dimension=1, option_type="put")
        v = eq.exact_solution(0.0, S0).item()
        assert 0.0 < v < 100.0

    def test_exact_gradient_matches_autograd(self):
        eq = XVAEquation(dimension=1)
        S = torch.tensor([[85.0], [100.0], [125.0]], requires_grad=True)
        u = eq.exact_solution(0.3, S)
        grad = torch.autograd.grad(u.sum(), S)[0]
        assert torch.allclose(grad, eq.exact_gradient(0.3, S.detach()), atol=1e-4)

    def test_multi_asset_has_no_closed_form(self):
        eq = XVAEquation(dimension=2)
        assert eq.exact_solution(0.0, torch.ones(1, 2) * 100.0) is None
        assert not eq.has_exact_solution()

    def test_bs_value_tensor_tau_matches_scalar(self):
        eq = XVAEquation(dimension=1)
        S = torch.tensor([[90.0], [110.0]])
        tau_tensor = torch.full((2, 1), 0.4)
        assert torch.allclose(eq._bs_value(0.4, S), eq._bs_value(tau_tensor, S))


def test_option_type_is_validated():
    with pytest.raises(ValueError, match="option_type"):
        XVAEquation(dimension=1, option_type="straddle")
