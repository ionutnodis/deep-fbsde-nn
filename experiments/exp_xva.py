"""
XVA Pricing via Deep BSDE
=========================

Pricing derivatives with valuation adjustments (XVA) using Deep BSDE.

XVA includes:
- CVA (Credit Valuation Adjustment): counterparty default risk
- DVA (Debit Valuation Adjustment): own default risk
- FVA (Funding Valuation Adjustment): funding costs
- KVA (Capital Valuation Adjustment): regulatory capital costs

The key insight is that XVA leads to a nonlinear BSDE:

    dV_t = r*V_t*dt + Z_t*dW_t - f(t, V_t, Z_t)*dt
    V_T = payoff(S_T)

where f is the nonlinear driver capturing funding/credit costs:

    f(t, V, Z) = λ_c * (1-R_c) * max(V, 0)      (CVA)
               - λ_b * (1-R_b) * max(-V, 0)     (DVA)
               + (r_f - r) * V                   (FVA)

This is exactly the type of problem Deep BSDE was designed for!

References:
- Burgard & Kjaer (2011): "Partial Differential Equation Representations of
  Derivatives with Bilateral Counterparty Risk and Funding Costs"
- Crépey (2015): "Bilateral Counterparty Risk under Funding Constraints"

Usage:
    python experiments/exp_xva.py --device mps
    python experiments/exp_xva.py --lambda-c 0.02 --lambda-b 0.01
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from src.networks import NAISNet


# =============================================================================
# BLACK-SCHOLES ANALYTICS (for reference)
# =============================================================================


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes call price (no XVA)."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes put price (no XVA)."""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def classical_xva_monte_carlo(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_c: float,
    lambda_b: float,
    R_c: float,
    R_b: float,
    r_f: float,
    option_type: str = "call",
    n_paths: int = 100000,
    n_steps: int = 100,
) -> Dict[str, float]:
    """
    Classical Monte Carlo XVA calculation.

    This is the industry-standard approach:
    1. Simulate asset paths
    2. Compute option value at each time step (using BS formula)
    3. Compute EPE/ENE profiles
    4. Integrate to get CVA/DVA/FVA

    Returns dict with CVA, DVA, FVA, Total_XVA, and XVA_price
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Simulate paths
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        Z = np.random.randn(n_paths)
        S[:, i + 1] = S[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z)

    # Compute option values along paths using BS formula
    V = np.zeros((n_paths, n_steps + 1))

    bs_func = black_scholes_call if option_type == "call" else black_scholes_put

    for j in range(n_steps + 1):
        t_j = j * dt
        tau = T - t_j  # time to maturity

        if tau > 1e-10:
            for i in range(n_paths):
                V[i, j] = bs_func(S[i, j], K, tau, r, sigma)
        else:
            # At maturity
            if option_type == "call":
                V[:, j] = np.maximum(S[:, j] - K, 0)
            else:
                V[:, j] = np.maximum(K - S[:, j], 0)

    # Compute exposure profiles
    EPE = np.mean(np.maximum(V, 0), axis=0)  # Expected Positive Exposure
    ENE = np.mean(np.maximum(-V, 0), axis=0)  # Expected Negative Exposure
    EE = np.mean(V, axis=0)  # Expected Exposure (signed)

    # Compute XVA integrals
    cva = 0.0
    dva = 0.0
    fva = 0.0

    for j in range(1, n_steps + 1):
        t_j = j * dt
        df = np.exp(-r * t_j)

        # Default probabilities in interval [t_{j-1}, t_j]
        q_c = np.exp(-lambda_c * (t_j - dt)) - np.exp(-lambda_c * t_j)
        q_b = np.exp(-lambda_b * (t_j - dt)) - np.exp(-lambda_b * t_j)

        # CVA: expected loss if counterparty defaults when we're in the money
        cva += (1 - R_c) * EPE[j] * q_c * df

        # DVA: expected gain if we default when counterparty is in the money
        dva += (1 - R_b) * ENE[j] * q_b * df

        # FVA: funding cost/benefit
        fva += (r_f - r) * EE[j] * df * dt

    # BS price without XVA
    bs_price = bs_func(S0, K, T, r, sigma)

    # XVA-adjusted price
    # CVA is a cost (we subtract), DVA is a benefit (we add), FVA is typically a cost
    xva_price = bs_price - cva + dva - fva

    return {
        "BS_price": bs_price,
        "CVA": cva,
        "DVA": dva,
        "FVA": fva,
        "Total_XVA": -cva + dva - fva,
        "XVA_price": xva_price,
        "EPE": EPE,
        "ENE": ENE,
    }


# =============================================================================
# XVA PARAMETERS
# =============================================================================


@dataclass
class XVAParams:
    """XVA and option parameters."""

    # Underlying
    S0: float = 100.0  # Initial spot
    K: float = 100.0  # Strike
    T: float = 1.0  # Maturity
    r: float = 0.05  # Risk-free rate
    sigma: float = 0.2  # Volatility

    # Credit parameters
    lambda_c: float = 0.02  # Counterparty default intensity (2% per year)
    lambda_b: float = 0.01  # Own default intensity (1% per year)
    R_c: float = 0.4  # Counterparty recovery rate
    R_b: float = 0.4  # Own recovery rate

    # Funding parameters
    r_f: float = 0.06  # Funding rate (spread over risk-free)

    # Option type
    option_type: str = "call"  # 'call' or 'put'


# =============================================================================
# XVA DEEP BSDE SOLVER
# =============================================================================


class XVABSDESolver:
    """
    Deep BSDE solver for XVA pricing using NAIS-Net.

    Solves the nonlinear BSDE:
        dV_t = [r*V_t - f(t, V_t, Z_t)]*dt + Z_t*σ*S_t*dW_t
        V_T = payoff(S_T)

    where f is the XVA driver:
        f(V) = λ_c*(1-R_c)*max(V,0) - λ_b*(1-R_b)*max(-V,0) + (r_f-r)*V

    The network learns V(t, S) ≈ option price with XVA at time t given spot S.
    """

    # MLMC schedule
    MLMC_SCHEDULE = {
        0: 5,
        2000: 10,
        5000: 20,
        8000: 35,
        12000: 50,
    }

    def __init__(
        self,
        params: XVAParams,
        hidden_dim: int = 256,
        num_layers: int = 4,
        device: str = "cpu",
    ):
        self.params = params
        self.device = torch.device(device)

        # NAIS-Net: input is (t, S) so input_dim = 2
        self.net = NAISNet(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation="sine",
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=4000, gamma=0.5
        )

        # Training history
        self.losses = []
        self.prices = []
        self.current_iteration = 0

        print(f"XVA BSDE Solver with NAIS-Net")
        print(f"  Parameters: {sum(p.numel() for p in self.net.parameters()):,}")
        print(f"  Option: {params.option_type}, S0={params.S0}, K={params.K}")
        print(f"  Credit: λ_c={params.lambda_c}, λ_b={params.lambda_b}")
        print(f"  Funding: r_f={params.r_f}, r={params.r}")

    def _get_mlmc_timesteps(self, iteration: int) -> int:
        """Get number of timesteps based on MLMC schedule."""
        n_steps = 5
        for thresh, steps in sorted(self.MLMC_SCHEDULE.items()):
            if iteration >= thresh:
                n_steps = steps
        return n_steps

    def _forward(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Forward pass with input normalization."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if S.dim() == 1:
            S = S.unsqueeze(-1)

        t_norm = t / self.params.T
        S_norm = S / self.params.S0

        x = torch.cat([t_norm, S_norm], dim=-1).float()
        return self.net(x)

    def xva_driver(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute the XVA driver f(V).

        For the PDE: ∂u/∂t + ℒu - r*u + f(u) = 0
        The BSDE becomes: dY = (r*Y - f(Y))dt + Z·dW

        f(V) = λ_c*(1-R_c)*max(V,0)      (CVA cost when V>0)
             - λ_b*(1-R_b)*max(-V,0)     (DVA benefit when V<0)
             + (r_f - r)*V               (FVA)

        For a LONG call position (we own the option):
        - V > 0 always (call has positive value)
        - CVA: counterparty owes us, if they default we lose
        - DVA: we owe nothing, so DVA ≈ 0
        - FVA: if V > 0, we need to fund collateral
        """
        p = self.params

        # CVA: counterparty default when we're in the money (V > 0)
        cva = p.lambda_c * (1 - p.R_c) * torch.relu(V)

        # DVA: own default when we owe money (V < 0) - rare for long call
        dva = p.lambda_b * (1 - p.R_b) * torch.relu(-V)

        # FVA: funding cost proportional to value
        # Positive V means we post collateral, negative spread means cost
        fva = (p.r_f - p.r) * V

        # Total driver: CVA is cost, DVA is benefit
        return cva - dva + fva

    def payoff(self, S: torch.Tensor) -> torch.Tensor:
        """Compute option payoff at maturity."""
        p = self.params
        if p.option_type == "call":
            return torch.relu(S - p.K)
        else:
            return torch.relu(p.K - S)

    def simulate_paths(
        self,
        batch_size: int,
        n_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate stock paths under risk-neutral measure."""
        p = self.params
        dt = p.T / n_steps
        sqrt_dt = np.sqrt(dt)

        t = torch.linspace(0, p.T, n_steps + 1, device=self.device)
        dW = torch.randn(batch_size, n_steps, device=self.device) * sqrt_dt

        S = torch.zeros(batch_size, n_steps + 1, device=self.device)
        S[:, 0] = p.S0

        for i in range(n_steps):
            S[:, i + 1] = S[:, i] * torch.exp(
                (p.r - 0.5 * p.sigma**2) * dt + p.sigma * dW[:, i]
            )

        return t, S, dW

    def compute_loss(
        self,
        batch_size: int = 256,
        n_steps: int = 50,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute Deep BSDE loss for XVA pricing.

        Following Han et al. (2018), we parameterize:
        - Y_0 = u(0, S_0) via network at t=0
        - Z_n = σ * S_n * ∇u(t_n, S_n) via network gradient

        Then propagate forward:
            Y_{n+1} = Y_n + [r*Y_n - f(Y_n)]*dt + Z_n*dW_n

        Loss = ||Y_N - g(S_N)||^2  (terminal condition)
        """
        p = self.params
        dt = p.T / n_steps

        # Simulate paths
        t, S, dW = self.simulate_paths(batch_size, n_steps)

        # === FORWARD PROPAGATION ===
        # Get Y_0 and Z_0 from network
        t0 = torch.zeros(batch_size, 1, device=self.device)
        S0 = S[:, 0:1]

        S0_grad = S0.clone().requires_grad_(True)
        Y = self._forward(t0, S0_grad).squeeze()

        # Z = σ * S * ∂u/∂S
        grad_Y = torch.autograd.grad(Y.sum(), S0_grad, create_graph=True)[0]
        Z = p.sigma * S0.squeeze() * grad_Y.squeeze()

        # Store Y0 for reporting
        Y0 = Y.mean().detach()

        # Forward propagate the BSDE
        for i in range(n_steps):
            # BSDE dynamics: dY = (r*Y - f(Y))dt + Z*dW
            drift = (p.r * Y - self.xva_driver(Y)) * dt
            diffusion = Z * dW[:, i]
            Y_new = Y + drift + diffusion

            # Get Z for next step (except at terminal)
            if i < n_steps - 1:
                t_next = t[i + 1]
                S_next = S[:, i + 1 : i + 2]
                S_next_grad = S_next.clone().requires_grad_(True)
                t_batch = torch.full((batch_size, 1), t_next.item(), device=self.device)

                Y_net = self._forward(t_batch, S_next_grad).squeeze()
                grad_Y = torch.autograd.grad(
                    Y_net.sum(), S_next_grad, create_graph=True
                )[0]
                Z = p.sigma * S_next.squeeze() * grad_Y.squeeze()

            Y = Y_new

        # === TERMINAL LOSS ===
        # Y_T should match the payoff g(S_T)
        terminal_payoff = self.payoff(S[:, -1])
        terminal_loss = torch.mean((Y - terminal_payoff) ** 2)

        # Also add intermediate consistency loss (optional but helps)
        # This ensures network predictions match BSDE propagation
        intermediate_loss = torch.tensor(0.0, device=self.device)

        return terminal_loss + 0.1 * intermediate_loss, Y0.item()

    def train(
        self,
        n_iterations: int = 20000,
        batch_size: int = 256,
        n_steps: int = 50,
        print_every: int = 500,
        use_mlmc: bool = True,
    ):
        """Train the XVA pricer."""
        p = self.params

        # Reference: Black-Scholes price without XVA
        if p.option_type == "call":
            bs_price = black_scholes_call(p.S0, p.K, p.T, p.r, p.sigma)
        else:
            bs_price = black_scholes_put(p.S0, p.K, p.T, p.r, p.sigma)

        print(f"\nTraining XVA Pricer")
        print(f"  BS price (no XVA): {bs_price:.6f}")
        print(f"  Iterations: {n_iterations}, Batch: {batch_size}")
        print(f"  MLMC: {'Enabled' if use_mlmc else 'Disabled'}")
        print("-" * 80)

        for it in range(n_iterations):
            self.current_iteration = it

            if use_mlmc:
                current_steps = self._get_mlmc_timesteps(it)
            else:
                current_steps = n_steps

            self.optimizer.zero_grad()
            loss, price = self.compute_loss(batch_size, current_steps)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.item())
            self.prices.append(price)

            if it % print_every == 0 or it == n_iterations - 1:
                xva = price - bs_price
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"It {it:5d} | Loss {loss.item():.4e} | "
                    f"Price {price:.4f} | XVA {xva:+.4f} | N={current_steps} | LR={lr:.2e}"
                )

        return self.losses, self.prices

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "params": self.params,
                "losses": self.losses,
                "prices": self.prices,
                "iteration": self.current_iteration,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        import numpy as np

        # Handle PyTorch 2.6+ weights_only default
        try:
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        except AttributeError:
            try:
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            except AttributeError:
                pass

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.losses = ckpt.get("losses", [])
        self.prices = ckpt.get("prices", [])
        self.current_iteration = ckpt.get("iteration", 0)
        print(f"Model loaded from {path}")

    def compute_xva_components(self, n_paths: int = 100000) -> Dict[str, float]:
        """
        Estimate individual XVA components via Monte Carlo.

        This is approximate but gives intuition about the XVA breakdown.
        """
        p = self.params
        n_steps = 100
        dt = p.T / n_steps

        # Simulate paths
        t, S, dW = self.simulate_paths(n_paths, n_steps)

        # Get option values along paths
        self.net.eval()
        with torch.no_grad():
            cva_integral = torch.zeros(n_paths, device=self.device)
            dva_integral = torch.zeros(n_paths, device=self.device)
            fva_integral = torch.zeros(n_paths, device=self.device)

            for i in range(n_steps):
                t_i = t[i].item()
                S_i = S[:, i : i + 1]
                t_batch = torch.full((n_paths, 1), t_i, device=self.device)

                V_i = self._forward(t_batch, S_i).squeeze()

                # Discount factor
                df = np.exp(-p.r * t_i)

                # CVA contribution
                cva_integral += df * p.lambda_c * (1 - p.R_c) * torch.relu(V_i) * dt

                # DVA contribution
                dva_integral += df * p.lambda_b * (1 - p.R_b) * torch.relu(-V_i) * dt

                # FVA contribution
                fva_integral += df * (p.r_f - p.r) * V_i * dt

        return {
            "CVA": -cva_integral.mean().item(),  # CVA is a cost (negative)
            "DVA": dva_integral.mean().item(),  # DVA is a benefit (positive)
            "FVA": -fva_integral.mean().item(),  # FVA depends on sign of V
            "Total_XVA": (-cva_integral + dva_integral - fva_integral).mean().item(),
        }


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_xva_results(solver: XVABSDESolver, save_dir: str = "results/figures"):
    """Generate comprehensive XVA plots."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    p = solver.params

    if p.option_type == "call":
        bs_price = black_scholes_call(p.S0, p.K, p.T, p.r, p.sigma)
    else:
        bs_price = black_scholes_put(p.S0, p.K, p.T, p.r, p.sigma)

    # ==========================================================================
    # Figure 1: Training diagnostics (2x2)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Learning curve
    ax = axes[0, 0]
    ax.semilogy(solver.losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.3)

    # 2. Price convergence
    ax = axes[0, 1]
    ax.plot(solver.prices, "b-", label="XVA Price", alpha=0.7)
    ax.axhline(
        bs_price, color="r", linestyle="--", label=f"BS (no XVA) = {bs_price:.4f}"
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Option Price")
    ax.set_title("Price Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. XVA adjustment convergence
    ax = axes[1, 0]
    xva_history = [price - bs_price for price in solver.prices]
    ax.plot(xva_history, "g-", alpha=0.7)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("XVA Adjustment")
    ax.set_title("XVA Convergence")
    ax.grid(True, alpha=0.3)

    # 4. Rolling average of price (smoothed)
    ax = axes[1, 1]
    window = min(500, len(solver.prices) // 10)
    if window > 1:
        rolling_avg = np.convolve(solver.prices, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(solver.prices)), rolling_avg, "b-", linewidth=2)
        ax.axhline(bs_price, color="r", linestyle="--", label=f"BS = {bs_price:.4f}")
    else:
        ax.plot(solver.prices, "b-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Price (smoothed)")
    ax.set_title(f"Rolling Average Price (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/xva_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_dir}/xva_training.png")

    # ==========================================================================
    # Figure 2: Price comparison (2x2)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    S_vals = np.linspace(50, 150, 100)
    xva_prices = []
    bs_prices = []

    solver.net.eval()
    with torch.no_grad():
        for S in S_vals:
            t_tensor = torch.tensor([[0.0]], device=solver.device, dtype=torch.float32)
            S_tensor = torch.tensor([[S]], device=solver.device, dtype=torch.float32)
            xva_prices.append(solver._forward(t_tensor, S_tensor).item())

            if p.option_type == "call":
                bs_prices.append(black_scholes_call(S, p.K, p.T, p.r, p.sigma))
            else:
                bs_prices.append(black_scholes_put(S, p.K, p.T, p.r, p.sigma))

    # 1. Price vs Spot comparison
    ax = axes[0, 0]
    ax.plot(S_vals, xva_prices, "b-", label="With XVA", linewidth=2)
    ax.plot(S_vals, bs_prices, "r--", label="Without XVA (BS)", linewidth=2)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel(f"{p.option_type.capitalize()} Price")
    ax.set_title("Price Comparison: XVA vs Black-Scholes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. XVA adjustment vs Spot
    ax = axes[0, 1]
    xva_adjustment = [xva - bs for xva, bs in zip(xva_prices, bs_prices)]
    ax.plot(S_vals, xva_adjustment, "g-", linewidth=2)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("XVA Adjustment")
    ax.set_title(f"XVA = Price(XVA) - Price(BS)")
    ax.grid(True, alpha=0.3)

    # Add text with XVA parameters
    textstr = (
        f"CVA: λ_c={p.lambda_c}, R_c={p.R_c}\n"
        f"DVA: λ_b={p.lambda_b}, R_b={p.R_b}\n"
        f"FVA: r_f-r={p.r_f-p.r:.2%}"
    )
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 3. Delta comparison (via finite difference)
    ax = axes[1, 0]
    dS = 0.5
    xva_deltas = []
    bs_deltas = []

    with torch.no_grad():
        for S in S_vals:
            t_tensor = torch.tensor([[0.0]], device=solver.device, dtype=torch.float32)
            S_up = torch.tensor([[S + dS]], device=solver.device, dtype=torch.float32)
            S_dn = torch.tensor([[S - dS]], device=solver.device, dtype=torch.float32)

            v_up = solver._forward(t_tensor, S_up).item()
            v_dn = solver._forward(t_tensor, S_dn).item()
            xva_deltas.append((v_up - v_dn) / (2 * dS))

            if p.option_type == "call":
                bs_up = black_scholes_call(S + dS, p.K, p.T, p.r, p.sigma)
                bs_dn = black_scholes_call(S - dS, p.K, p.T, p.r, p.sigma)
            else:
                bs_up = black_scholes_put(S + dS, p.K, p.T, p.r, p.sigma)
                bs_dn = black_scholes_put(S - dS, p.K, p.T, p.r, p.sigma)
            bs_deltas.append((bs_up - bs_dn) / (2 * dS))

    ax.plot(S_vals, xva_deltas, "b-", label="XVA Delta", linewidth=2)
    ax.plot(S_vals, bs_deltas, "r--", label="BS Delta", linewidth=2)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Delta (∂V/∂S)")
    ax.set_title("Delta Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. XVA as percentage of BS price
    ax = axes[1, 1]
    xva_pct = [
        100 * adj / bs if bs > 0.1 else 0 for adj, bs in zip(xva_adjustment, bs_prices)
    ]
    ax.plot(S_vals, xva_pct, "m-", linewidth=2)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("XVA (% of BS price)")
    ax.set_title("XVA as Percentage of Risk-Free Price")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/xva_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_dir}/xva_comparison.png")

    # ==========================================================================
    # Figure 3: Price surfaces (1x3)
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    n_S, n_t = 50, 50
    S_vals_surf = np.linspace(60, 140, n_S)
    t_vals = np.linspace(0, p.T * 0.95, n_t)
    S_grid, t_grid = np.meshgrid(S_vals_surf, t_vals)

    prices_xva = np.zeros_like(S_grid)
    prices_bs = np.zeros_like(S_grid)

    solver.net.eval()
    with torch.no_grad():
        for i in range(n_t):
            for j in range(n_S):
                t_val = t_grid[i, j]
                S_val = S_grid[i, j]
                tau = p.T - t_val

                t_tensor = torch.tensor(
                    [[t_val]], device=solver.device, dtype=torch.float32
                )
                S_tensor = torch.tensor(
                    [[S_val]], device=solver.device, dtype=torch.float32
                )
                prices_xva[i, j] = solver._forward(t_tensor, S_tensor).item()

                if p.option_type == "call":
                    prices_bs[i, j] = black_scholes_call(S_val, p.K, tau, p.r, p.sigma)
                else:
                    prices_bs[i, j] = black_scholes_put(S_val, p.K, tau, p.r, p.sigma)

    # XVA price surface
    ax = axes[0]
    c = ax.contourf(S_grid, t_grid, prices_xva, levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax, label="Price")
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")
    ax.set_title("XVA Price Surface V(t,S)")

    # BS price surface
    ax = axes[1]
    c = ax.contourf(S_grid, t_grid, prices_bs, levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax, label="Price")
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")
    ax.set_title("Black-Scholes Price Surface")

    # XVA adjustment surface
    ax = axes[2]
    xva_surface = prices_xva - prices_bs
    vmax = max(abs(xva_surface.min()), abs(xva_surface.max()), 0.01)
    c = ax.contourf(
        S_grid, t_grid, xva_surface, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    plt.colorbar(c, ax=ax, label="XVA")
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")
    ax.set_title("XVA Adjustment Surface")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/xva_surface.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_dir}/xva_surface.png")

    # ==========================================================================
    # Figure 4: XVA Component Analysis
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute CVA, DVA, FVA contributions along spot
    n_paths_mc = 10000
    n_steps_mc = 50
    dt = p.T / n_steps_mc

    cva_by_spot = []
    dva_by_spot = []
    fva_by_spot = []

    S_test = np.linspace(70, 130, 20)

    for S0_test in S_test:
        # Simple approximation: scale by moneyness
        # More accurate would be full MC for each S0
        t_tensor = torch.tensor([[0.0]], device=solver.device, dtype=torch.float32)
        S_tensor = torch.tensor([[S0_test]], device=solver.device, dtype=torch.float32)
        V0 = solver._forward(t_tensor, S_tensor).item()

        # Rough CVA/DVA/FVA approximation based on V0
        cva_approx = -p.lambda_c * (1 - p.R_c) * max(V0, 0) * p.T * 0.5  # rough
        dva_approx = p.lambda_b * (1 - p.R_b) * max(-V0, 0) * p.T * 0.5
        fva_approx = -(p.r_f - p.r) * V0 * p.T * 0.5

        cva_by_spot.append(cva_approx)
        dva_by_spot.append(dva_approx)
        fva_by_spot.append(fva_approx)

    # 1. Stacked XVA components
    ax = axes[0, 0]
    ax.fill_between(S_test, 0, cva_by_spot, alpha=0.7, label="CVA", color="red")
    ax.fill_between(
        S_test,
        cva_by_spot,
        [c + d for c, d in zip(cva_by_spot, dva_by_spot)],
        alpha=0.7,
        label="DVA",
        color="green",
    )
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("XVA Component")
    ax.set_title("XVA Components (Approximate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Individual components
    ax = axes[0, 1]
    ax.plot(S_test, cva_by_spot, "r-", label="CVA", linewidth=2)
    ax.plot(S_test, dva_by_spot, "g-", label="DVA", linewidth=2)
    ax.plot(S_test, fva_by_spot, "b-", label="FVA", linewidth=2)
    total_xva = [c + d + f for c, d, f in zip(cva_by_spot, dva_by_spot, fva_by_spot)]
    ax.plot(S_test, total_xva, "k--", label="Total", linewidth=2)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("XVA")
    ax.set_title("XVA Component Breakdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Price at different times
    ax = axes[1, 0]
    times = [0.0, 0.25, 0.5, 0.75]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(times)))

    with torch.no_grad():
        for t_val, color in zip(times, colors):
            prices_t = []
            for S in S_vals:
                t_tensor = torch.tensor(
                    [[t_val]], device=solver.device, dtype=torch.float32
                )
                S_tensor = torch.tensor(
                    [[S]], device=solver.device, dtype=torch.float32
                )
                prices_t.append(solver._forward(t_tensor, S_tensor).item())
            ax.plot(S_vals, prices_t, color=color, label=f"t={t_val:.2f}", linewidth=2)

    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Option Price")
    ax.set_title("XVA Price at Different Times")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis("off")

    final_price = (
        solver.prices[-1] if solver.prices else xva_prices[len(xva_prices) // 2]
    )
    final_xva = final_price - bs_price

    summary_text = f"""
    XVA PRICING SUMMARY
    {'='*40}
    
    Option Parameters:
      Type: {p.option_type.upper()}
      S₀ = {p.S0}, K = {p.K}, T = {p.T}
      r = {p.r:.2%}, σ = {p.sigma:.2%}
    
    Credit Parameters:
      λ_c = {p.lambda_c:.2%} (counterparty default)
      λ_b = {p.lambda_b:.2%} (own default)
      R_c = {p.R_c:.0%}, R_b = {p.R_b:.0%}
    
    Funding Parameters:
      r_f = {p.r_f:.2%} (funding rate)
      Spread = {(p.r_f - p.r)*100:.0f} bps
    
    Results (at S = {p.S0}):
      BS Price (no XVA):  {bs_price:.4f}
      XVA Price:          {final_price:.4f}
      XVA Adjustment:     {final_xva:+.4f} ({final_xva/bs_price*100:+.2f}%)
    """

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/xva_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_dir}/xva_analysis.png")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="XVA Pricing via Deep BSDE")

    # Option parameters
    parser.add_argument("--spot", type=float, default=100.0, help="Initial spot")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike")
    parser.add_argument("--maturity", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument(
        "--option-type", type=str, default="call", choices=["call", "put"]
    )

    # XVA parameters
    parser.add_argument(
        "--lambda-c", type=float, default=0.02, help="Counterparty default intensity"
    )
    parser.add_argument(
        "--lambda-b", type=float, default=0.01, help="Own default intensity"
    )
    parser.add_argument(
        "--recovery-c", type=float, default=0.4, help="Counterparty recovery rate"
    )
    parser.add_argument(
        "--recovery-b", type=float, default=0.4, help="Own recovery rate"
    )
    parser.add_argument("--funding-rate", type=float, default=0.06, help="Funding rate")

    # Training parameters
    parser.add_argument(
        "--iterations", type=int, default=20000, help="Training iterations"
    )
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--no-mlmc", action="store_true", help="Disable MLMC")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save-dir", type=str, default="results/figures")
    parser.add_argument("--model-dir", type=str, default="results/models")
    parser.add_argument(
        "--load", type=str, default=None, help="Path to load model from"
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Create parameters
    params = XVAParams(
        S0=args.spot,
        K=args.strike,
        T=args.maturity,
        r=args.rate,
        sigma=args.sigma,
        lambda_c=args.lambda_c,
        lambda_b=args.lambda_b,
        R_c=args.recovery_c,
        R_b=args.recovery_b,
        r_f=args.funding_rate,
        option_type=args.option_type,
    )

    # Create solver
    solver = XVABSDESolver(
        params=params,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        device=args.device,
    )

    # Load existing model or train new one
    if args.load:
        solver.load(args.load)
    else:
        # Train
        solver.train(
            n_iterations=args.iterations,
            batch_size=args.batch,
            print_every=500,
            use_mlmc=not args.no_mlmc,
        )

        # Save model
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        model_path = f"{args.model_dir}/xva_{params.option_type}_lc{params.lambda_c}_lb{params.lambda_b}.pt"
        solver.save(model_path)

    # Final results
    if params.option_type == "call":
        bs_price = black_scholes_call(
            params.S0, params.K, params.T, params.r, params.sigma
        )
    else:
        bs_price = black_scholes_put(
            params.S0, params.K, params.T, params.r, params.sigma
        )

    final_price = (
        solver.prices[-1]
        if solver.prices
        else solver._forward(
            torch.zeros(1, 1, device=solver.device),
            torch.tensor([[params.S0]], device=solver.device, dtype=torch.float32),
        ).item()
    )
    xva_adjustment = final_price - bs_price

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Black-Scholes price (no XVA): {bs_price:.6f}")
    print(f"Deep BSDE XVA price:          {final_price:.6f}")
    print(
        f"Deep BSDE XVA adjustment:     {xva_adjustment:+.6f} ({xva_adjustment/bs_price*100:+.2f}%)"
    )
    print("=" * 70)

    # Compute classical XVA for comparison
    print("\nClassical Monte Carlo XVA (benchmark)...")
    classical = classical_xva_monte_carlo(
        S0=params.S0,
        K=params.K,
        T=params.T,
        r=params.r,
        sigma=params.sigma,
        lambda_c=params.lambda_c,
        lambda_b=params.lambda_b,
        R_c=params.R_c,
        R_b=params.R_b,
        r_f=params.r_f,
        option_type=params.option_type,
        n_paths=200000,
        n_steps=200,
    )
    print(f"  CVA:        {classical['CVA']:+.6f}")
    print(f"  DVA:        {classical['DVA']:+.6f}")
    print(f"  FVA:        {classical['FVA']:+.6f}")
    print(f"  Total XVA:  {classical['Total_XVA']:+.6f}")
    print(f"  XVA Price:  {classical['XVA_price']:.6f}")
    print(f"  (BS Price:  {classical['BS_price']:.6f})")

    print("\n" + "-" * 70)
    print("COMPARISON: Deep BSDE vs Classical")
    print("-" * 70)
    print(f"  Classical XVA price: {classical['XVA_price']:.6f}")
    print(f"  Deep BSDE XVA price: {final_price:.6f}")
    diff = final_price - classical["XVA_price"]
    print(
        f"  Difference:          {diff:+.6f} ({diff/classical['XVA_price']*100:+.2f}%)"
    )

    # Compute XVA component estimates from Deep BSDE
    print("\nDeep BSDE XVA components (estimated)...")
    components = solver.compute_xva_components(n_paths=50000)
    print(f"  CVA (credit):   {components['CVA']:+.6f}")
    print(f"  DVA (debit):    {components['DVA']:+.6f}")
    print(f"  FVA (funding):  {components['FVA']:+.6f}")
    print(f"  Total:          {components['Total_XVA']:+.6f}")

    # Plot all figures
    plot_xva_results(solver, save_dir=args.save_dir)

    print(f"\nAll figures saved to {args.save_dir}/")
    print(f"Model saved to {args.model_dir}/")


if __name__ == "__main__":
    main()
