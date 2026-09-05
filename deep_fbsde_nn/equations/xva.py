"""
XVA Pricing Equation (Burgard-Kjaer)
====================================

Option pricing with valuation adjustments — counterparty credit (CVA), own
credit (DVA), and funding (FVA) — as a nonlinear BSDE.

PDE (library form: ∂u/∂t + μ·∇u + (1/2)Tr(σσᵀD²u) + f = 0):
    ∂V/∂t + rS·∇V + (1/2)σ²Tr(SSᵀ∘D²V) - rV - f_xva(V) = 0
    V(T, S) = payoff(S)

with the cost-positive XVA driver
    f_xva(V) = λ_c(1-R_c)·V⁺ - λ_b(1-R_b)·V⁻ + (r_f - r)·V

so ``driver()`` returns f = -rV - f_xva(V), and along paths the value
process gains (rV + f_xva(V))dt — costs reduce the initial price. With
λ_c = λ_b = 0 and r_f = r this reduces exactly to ``BlackScholesEquation``'s
driver (-rV).

Exact solution (long options, D=1): a long call or put has V ≥ 0 along every
path, so V⁻ ≡ 0 and the driver is LINEAR on the solution's range:
f_xva(V) = c·V with c = λ_c(1-R_c) + (r_f - r). The exact XVA-adjusted value
is therefore
    V(t, S) = e^{-c(T-t)} · BS(t, S)
— implemented in-class and cross-validated against a classical Monte-Carlo
XVA engine (exposure-profile integration) in the test suite.

References:
- Burgard & Kjaer (2011): "Partial Differential Equation Representations of
  Derivatives with Bilateral Counterparty Risk and Funding Costs"
- Crépey (2015): "Bilateral Counterparty Risk under Funding Constraints"
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseEquation, EquationConfig
from .black_scholes import _norm_cdf


class XVAEquation(BaseEquation):
    """
    XVA-adjusted option pricing under GBM dynamics.

    Args:
        dimension: Number of underlyings (payoff is on the basket sum;
            the closed-form ``exact_solution`` applies at D=1)
        S0: Initial spot per asset
        strike: Strike K (default: S0 * dimension, ATM on the basket)
        r: Risk-free rate
        sigma: Volatility
        lambda_c / lambda_b: Counterparty / own default intensities
        R_c / R_b: Counterparty / own recovery rates
        r_f: Funding rate (r_f - r is the funding spread)
        option_type: 'call' or 'put' (long positions; V >= 0)
        terminal_time: Maturity T

    Example:
        >>> eq = XVAEquation(dimension=1)          # Burgard-Kjaer defaults
        >>> eq.exact_solution(0.0, torch.tensor([[100.0]]))
        tensor([[10.2232]])                         # e^{-cT} * BS = 10.2232
    """

    def __init__(
        self,
        dimension: int = 1,
        S0: float = 100.0,
        strike: Optional[float] = None,
        r: float = 0.05,
        sigma: float = 0.2,
        lambda_c: float = 0.02,
        lambda_b: float = 0.01,
        R_c: float = 0.4,
        R_b: float = 0.4,
        r_f: float = 0.06,
        option_type: str = "call",
        terminal_time: float = 1.0,
        device: torch.device = None,
    ):
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
        config = EquationConfig(
            name=f"XVA ({option_type})",
            dimension=dimension,
            terminal_time=terminal_time,
        )
        super().__init__(config, device)

        self.S0 = S0
        self.strike = float(S0 * dimension) if strike is None else float(strike)
        self.r = r
        self._sigma_val = sigma
        self.lambda_c = lambda_c
        self.lambda_b = lambda_b
        self.R_c = R_c
        self.R_b = R_b
        self.r_f = r_f
        self.option_type = option_type

    # ------------------------------------------------------------------ SDE
    def drift(self, t, X, Y, Z):
        """μ = rX (risk-neutral GBM)."""
        return self.r * X

    def diffusion(self, t, X, Y):
        """σ = σ·X (diagonal)."""
        return self._sigma_val * X

    # ---------------------------------------------------------------- driver
    def xva_costs(self, V: torch.Tensor) -> torch.Tensor:
        """The cost-positive Burgard-Kjaer adjustment f_xva(V)."""
        cva = self.lambda_c * (1.0 - self.R_c) * torch.relu(V)
        dva = self.lambda_b * (1.0 - self.R_b) * torch.relu(-V)
        fva = (self.r_f - self.r) * V
        return cva - dva + fva

    def driver(self, t, X, Y, Z):
        """f = -rY - f_xva(Y): dY = (rY + f_xva(Y))dt + Zᵀσ dW along paths.

        Reduces to BlackScholesEquation's -rY when all adjustments vanish.
        """
        return -self.r * Y - self.xva_costs(Y)

    # -------------------------------------------------------------- terminal
    def terminal(self, X: torch.Tensor) -> torch.Tensor:
        basket = X.sum(dim=1, keepdim=True)
        if self.option_type == "call":
            return F.relu(basket - self.strike)
        return F.relu(self.strike - basket)

    def terminal_gradient(self, X: torch.Tensor) -> torch.Tensor:
        basket = X.sum(dim=1, keepdim=True)
        if self.option_type == "call":
            indicator = (basket > self.strike).float()
            return torch.ones_like(X) * indicator
        indicator = (basket < self.strike).float()
        return -torch.ones_like(X) * indicator

    def sample_initial_condition(self, batch_size: int = 1) -> torch.Tensor:
        return torch.full((batch_size, self.D), self.S0, device=self.device)

    # ----------------------------------------------------------------- exact
    @property
    def effective_cost_rate(self) -> float:
        """c = λ_c(1-R_c) + (r_f - r): the linearized cost rate on V ≥ 0."""
        return self.lambda_c * (1.0 - self.R_c) + (self.r_f - self.r)

    def _bs_value(self, tau, S: torch.Tensor) -> torch.Tensor:
        """Black-Scholes value of the (long, D=1) option.

        ``tau`` may be a float or a per-row tensor (broadcastable with S) —
        tensor support makes this usable as a differentiable analytic
        reference for control-variate surface learning.
        """
        K = self.strike
        if not torch.is_tensor(tau):
            tau = torch.full_like(S, float(tau))
        payoff = F.relu(S - K) if self.option_type == "call" else F.relu(K - S)
        tau_safe = tau.clamp_min(1e-8)
        sqrt_tau = torch.sqrt(tau_safe)
        d1 = (torch.log(S / K) + (self.r + 0.5 * self._sigma_val**2) * tau_safe) / (
            self._sigma_val * sqrt_tau
        )
        d2 = d1 - self._sigma_val * sqrt_tau
        discount = torch.exp(-self.r * tau_safe)
        call = S * _norm_cdf(d1) - K * discount * _norm_cdf(d2)
        value = call if self.option_type == "call" else call - S + K * discount
        return torch.where(tau <= 1e-8, payoff, value)

    def exact_solution(self, t: float, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        e^{-c(T-t)} · BS(t, S) — exact for D=1 long options.

        Long calls and puts have V ≥ 0 on every path, so the Burgard-Kjaer
        driver is linear (f_xva = c·V) on the solution's range and the
        nonlinear BSDE solves in closed form. Returns None for D > 1
        (basket options have no closed form; use a Monte-Carlo benchmark).
        """
        if self.D != 1:
            return None
        tau = self.T - float(t)
        S = X[:, :1]
        return float(np.exp(-self.effective_cost_rate * tau)) * self._bs_value(tau, S)

    def exact_gradient(self, t: float, X: torch.Tensor) -> Optional[torch.Tensor]:
        """∂V/∂S of the exact solution (D=1): e^{-cτ} · BS delta."""
        if self.D != 1:
            return None
        tau = self.T - float(t)
        S = X[:, :1]
        if tau <= 0:
            basket_itm = (
                (S > self.strike) if self.option_type == "call" else (S < self.strike)
            )
            sign = 1.0 if self.option_type == "call" else -1.0
            return sign * basket_itm.float()
        sqrt_tau = np.sqrt(tau)
        d1 = (torch.log(S / self.strike) + (self.r + 0.5 * self._sigma_val**2) * tau) / (
            self._sigma_val * sqrt_tau
        )
        delta = _norm_cdf(d1)
        if self.option_type == "put":
            delta = delta - 1.0
        return float(np.exp(-self.effective_cost_rate * tau)) * delta

    def has_exact_solution(self) -> bool:
        return self.D == 1
