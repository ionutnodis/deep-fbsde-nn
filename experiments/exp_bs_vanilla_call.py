"""
European Call Option Pricing via Deep BSDE
===========================================

A clean implementation for pricing a single-asset European call option
using the Deep BSDE method with NAIS-Net architecture. This serves as a
sanity check since:

1. Closed-form Black-Scholes solution exists
2. Simple 1D problem (no curse of dimensionality)
3. Well-understood dynamics

The PDE (Black-Scholes):
    ∂u/∂t + r*S*∂u/∂S + ½σ²S²*∂²u/∂S² - r*u = 0
    u(T, S) = max(S - K, 0)

Equivalent BSDE:
    dY_t = r*Y_t*dt + Z_t*dW_t
    Y_T = max(S_T - K, 0)

where Y_t = e^{-r(T-t)} * u(t, S_t) is the discounted option price
and Z_t = e^{-r(T-t)} * σ * S_t * ∂u/∂S

Usage:
    python experiments/exp_european_call.py
    python experiments/exp_european_call.py --spot 100 --strike 100 --sigma 0.2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm

from deep_fbsde_nn.networks import NAISNet

# =============================================================================
# BLACK-SCHOLES ANALYTICS
# =============================================================================


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Analytical Black-Scholes delta."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# =============================================================================
# DEEP BSDE SOLVER FOR EUROPEAN CALL
# =============================================================================


@dataclass
class CallOptionParams:
    """European call option parameters."""

    S0: float = 100.0  # Initial spot
    K: float = 100.0  # Strike
    T: float = 1.0  # Maturity
    r: float = 0.05  # Risk-free rate
    sigma: float = 0.2  # Volatility


class EuropeanCallBSDESolver:
    """
    Deep BSDE solver for European call option using NAIS-Net.

    We solve:
        dS_t = r*S_t*dt + σ*S_t*dW_t  (risk-neutral dynamics)
        dY_t = r*Y_t*dt + Z_t*σ*dW_t  (BSDE for discounted price)
        Y_T = max(S_T - K, 0)         (terminal condition)

    The network learns u(t, S) ≈ call price at time t given spot S.

    Uses Multi-Level Monte Carlo (MLMC) for efficient training:
    - Start with few timesteps (coarse) for fast initial learning
    - Gradually increase timesteps (fine) for accuracy
    """

    # MLMC schedule: iteration -> number of timesteps
    MLMC_SCHEDULE = {
        0: 5,
        2000: 10,
        5000: 20,
        8000: 35,
        10000: 50,
    }

    def __init__(
        self,
        params: CallOptionParams,
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
            activation="silu",
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3000, gamma=0.5
        )
        self.losses = []
        self.prices = []
        self.current_iteration = 0
        print(f"NAIS-Net parameters: {sum(p.numel() for p in self.net.parameters()):,}")

    def _get_mlmc_timesteps(self, iteration: int) -> int:
        """Get number of timesteps based on MLMC schedule."""
        n_steps = 5  # default
        for thresh, steps in sorted(self.MLMC_SCHEDULE.items()):
            if iteration >= thresh:
                n_steps = steps
        return n_steps

    def simulate_paths(
        self,
        batch_size: int,
        n_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate stock paths under risk-neutral measure.

        Returns:
            t: Time grid, shape (n_steps + 1,)
            S: Stock paths, shape (batch_size, n_steps + 1)
            dW: Brownian increments, shape (batch_size, n_steps)
        """
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

    def _forward(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Log-Moneyness transformation.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if S.dim() == 1:
            S = S.unsqueeze(-1)

        # --- FIX 2: Log-Moneyness Transformation ---
        # Transform S -> log(S/K) to center ATM at 0
        log_moneyness = torch.log(S / self.params.K)

        # Transform t -> time to maturity
        tau = self.params.T - t

        # Concatenate and cast
        x = torch.cat([tau, log_moneyness], dim=-1).to(torch.float32)
        return self.net(x)

    def compute_loss(
        self,
        batch_size: int = 256,
        n_steps: int = 50,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute Deep BSDE loss.

        The loss penalizes:
        1. Deviation from BSDE dynamics at each timestep
        2. Mismatch at terminal condition
        """
        p = self.params
        dt = p.T / n_steps

        # Simulate paths
        t, S, dW = self.simulate_paths(batch_size, n_steps)

        # Get network prediction at t=0
        t0 = torch.zeros(batch_size, 1, device=self.device)
        S0 = S[:, 0:1]

        S0_grad = S0.clone().requires_grad_(True)
        Y0 = self._forward(t0, S0_grad).squeeze()

        # Compute Z0 = σ * S * ∂u/∂S (the hedge)
        grad_Y0 = torch.autograd.grad(Y0.sum(), S0_grad, create_graph=True)[0]
        Z0 = p.sigma * S0.squeeze() * grad_Y0.squeeze()

        # Forward pass through BSDE
        Y = Y0
        Z = Z0

        loss = torch.tensor(0.0, device=self.device)

        for i in range(n_steps):
            # Current values
            t[i]
            S[:, i]

            # BSDE update: dY = r*Y*dt + Z*dW
            # (under risk-neutral measure, drift of Y is r*Y)
            dY = p.r * Y * dt + Z * dW[:, i]
            Y_next_bsde = Y + dY

            # Network prediction at next timestep
            t_next = t[i + 1]
            S_next = S[:, i + 1 : i + 2]

            S_next_grad = S_next.clone().requires_grad_(True)
            t_next_batch = torch.full(
                (batch_size, 1), t_next.item(), device=self.device
            )
            Y_next_net = self._forward(t_next_batch, S_next_grad).squeeze()

            # Compute Z at next timestep
            if i < n_steps - 1:
                grad_Y_next = torch.autograd.grad(
                    Y_next_net.sum(), S_next_grad, create_graph=True
                )[0]
                Z_next = p.sigma * S_next.squeeze() * grad_Y_next.squeeze()

            # Loss: penalize deviation from BSDE dynamics
            loss = loss + torch.mean((Y_next_net - Y_next_bsde) ** 2)

            # Update for next iteration
            Y = Y_next_net
            if i < n_steps - 1:
                Z = Z_next

        # Terminal condition: Y_T should equal payoff
        payoff = torch.relu(S[:, -1] - p.K)
        terminal_loss = torch.mean((Y - payoff) ** 2)

        # Total loss - heavily weight terminal condition
        # The terminal condition is what we actually care about!
        total_loss = loss / n_steps + 100.0 * terminal_loss

        # Return predicted price at t=0
        with torch.no_grad():
            price = self._forward(
                torch.zeros(1, 1, device=self.device),
                torch.tensor([[p.S0]], device=self.device, dtype=torch.float32),
            ).item()

        return total_loss, price

    def train(
        self,
        n_iterations: int = 15000,
        batch_size: int = 1,
        n_steps: int = 50,  # Max timesteps (used at end of training)
        print_every: int = 500,
        use_mlmc: bool = True,
    ):
        """
        Train the network with optional MLMC.

        Args:
            n_iterations: Total training iterations
            batch_size: Batch size for MC paths
            n_steps: Maximum timesteps (only used if use_mlmc=False)
            print_every: Print frequency
            use_mlmc: Whether to use Multi-Level Monte Carlo schedule
        """
        p = self.params
        exact_price = black_scholes_call(p.S0, p.K, p.T, p.r, p.sigma)

        print("Training European Call Option Pricer (NAIS-Net)")
        print(f"  S0={p.S0}, K={p.K}, T={p.T}, r={p.r}, σ={p.sigma}")
        print(f"  Exact BS price: {exact_price:.6f}")
        print(f"  Iterations: {n_iterations}, Batch: {batch_size}")
        print(f"  MLMC: {'Enabled' if use_mlmc else 'Disabled'}")
        print("-" * 70)

        for it in range(n_iterations):
            self.current_iteration = it

            # Get timesteps from MLMC schedule
            if use_mlmc:
                current_steps = self._get_mlmc_timesteps(it)
            else:
                current_steps = n_steps

            self.optimizer.zero_grad()
            loss, price = self.compute_loss(batch_size, current_steps)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.item())
            self.prices.append(price)

            if it % print_every == 0 or it == n_iterations - 1:
                error = abs(price - exact_price) / exact_price * 100
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"It {it:5d} | Loss {loss.item():.4e} | "
                    f"Price {price:.4f} | Error {error:.2f}% | N={current_steps} | LR={lr:.2e}"
                )

        return self.losses, self.prices

    def price_surface(
        self,
        S_range: Tuple[float, float] = (50, 150),
        t_range: Tuple[float, float] = (0, 1),
        n_S: int = 50,
        n_t: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute price surface over (t, S) grid.

        Returns:
            S_grid, t_grid: Meshgrid arrays
            prices_nn: Neural network prices
            prices_bs: Black-Scholes prices
        """
        p = self.params

        S_vals = np.linspace(S_range[0], S_range[1], n_S)
        t_vals = np.linspace(t_range[0], t_range[1], n_t)
        S_grid, t_grid = np.meshgrid(S_vals, t_vals)

        prices_nn = np.zeros_like(S_grid)
        prices_bs = np.zeros_like(S_grid)

        self.net.eval()
        with torch.no_grad():
            for i in range(n_t):
                for j in range(n_S):
                    t_val = t_grid[i, j]
                    S_val = S_grid[i, j]
                    tau = p.T - t_val  # Time to maturity

                    # Neural network price
                    t_tensor = torch.tensor(
                        [[t_val]], device=self.device, dtype=torch.float32
                    )
                    S_tensor = torch.tensor(
                        [[S_val]], device=self.device, dtype=torch.float32
                    )
                    prices_nn[i, j] = self._forward(t_tensor, S_tensor).item()

                    # Black-Scholes price
                    prices_bs[i, j] = black_scholes_call(S_val, p.K, tau, p.r, p.sigma)

        return S_grid, t_grid, prices_nn, prices_bs


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_results(solver: EuropeanCallBSDESolver, save_dir: str = "results/figures"):
    """Generate all plots."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    p = solver.params
    exact_price = black_scholes_call(p.S0, p.K, p.T, p.r, p.sigma)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Learning curve
    ax = axes[0, 0]
    ax.semilogy(solver.losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.3)

    # 2. Price convergence
    ax = axes[0, 1]
    ax.plot(solver.prices, label="NN Price")
    ax.axhline(exact_price, color="r", linestyle="--", label=f"BS = {exact_price:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Option Price")
    ax.set_title("Price Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Price vs Spot at t=0
    ax = axes[1, 0]
    S_vals = np.linspace(50, 150, 100)
    nn_prices = []
    bs_prices = []

    solver.net.eval()
    with torch.no_grad():
        for S in S_vals:
            t_tensor = torch.tensor([[0.0]], device=solver.device)
            S_tensor = torch.tensor([[S]], device=solver.device)
            nn_prices.append(solver._forward(t_tensor, S_tensor).item())
            bs_prices.append(black_scholes_call(S, p.K, p.T, p.r, p.sigma))

    ax.plot(S_vals, nn_prices, "b-", label="Neural Network", linewidth=2)
    ax.plot(S_vals, bs_prices, "r--", label="Black-Scholes", linewidth=2)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5, label="Strike")
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Call Price")
    ax.set_title(f"Price vs Spot at t=0 (K={p.K})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Delta vs Spot at t=0
    ax = axes[1, 1]
    nn_deltas = []
    bs_deltas = []

    for S in S_vals:
        # NN delta via autograd
        t_tensor = torch.tensor([[0.0]], device=solver.device)
        S_tensor = torch.tensor([[S]], device=solver.device, requires_grad=True)
        price = solver._forward(t_tensor, S_tensor)
        price.backward()
        # Need to scale by normalization factor (1/S0) since we normalized input
        nn_deltas.append(S_tensor.grad.item() / solver.params.S0)

        # BS delta
        bs_deltas.append(black_scholes_delta(S, p.K, p.T, p.r, p.sigma))

    ax.plot(S_vals, nn_deltas, "b-", label="Neural Network", linewidth=2)
    ax.plot(S_vals, bs_deltas, "r--", label="Black-Scholes", linewidth=2)
    ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Delta")
    ax.set_title(f"Delta vs Spot at t=0 (K={p.K})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/european_call_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_dir}/european_call_results.png")

    # 5. Price surface comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    S_grid, t_grid, prices_nn, prices_bs = solver.price_surface()

    # NN surface
    ax = axes[0]
    c = ax.contourf(S_grid, t_grid, prices_nn, levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")
    ax.set_title("Neural Network Price")

    # BS surface
    ax = axes[1]
    c = ax.contourf(S_grid, t_grid, prices_bs, levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")
    ax.set_title("Black-Scholes Price")

    # Error
    ax = axes[2]
    error = np.abs(prices_nn - prices_bs)
    c = ax.contourf(S_grid, t_grid, error, levels=20, cmap="Reds")
    plt.colorbar(c, ax=ax)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")
    ax.set_title("Absolute Error")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/european_call_surface.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {save_dir}/european_call_surface.png")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="European Call via Deep BSDE")
    parser.add_argument("--spot", type=float, default=100.0, help="Initial spot")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike")
    parser.add_argument("--maturity", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument(
        "--iterations", type=int, default=15000, help="Training iterations"
    )
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--steps", type=int, default=50, help="Max time steps")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--no-mlmc", action="store_true", help="Disable MLMC")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save-dir", type=str, default="results/figures")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Option parameters
    params = CallOptionParams(
        S0=args.spot,
        K=args.strike,
        T=args.maturity,
        r=args.rate,
        sigma=args.sigma,
    )

    # Create solver with NAIS-Net
    solver = EuropeanCallBSDESolver(
        params=params,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        device=args.device,
    )

    # Train with MLMC
    solver.train(
        n_iterations=args.iterations,
        batch_size=args.batch,
        n_steps=args.steps,
        print_every=500,
        use_mlmc=not args.no_mlmc,
    )

    # Final results
    exact = black_scholes_call(params.S0, params.K, params.T, params.r, params.sigma)
    final_price = solver.prices[-1]
    final_error = abs(final_price - exact) / exact * 100

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Black-Scholes price: {exact:.6f}")
    print(f"Neural network price: {final_price:.6f}")
    print(f"Relative error: {final_error:.4f}%")

    # Plot
    plot_results(solver, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
