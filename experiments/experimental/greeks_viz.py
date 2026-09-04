"""
Greeks Visualization — EXPERIMENTAL
===================================

Price/Delta-vs-spot curves for Black-Scholes-type models, extracted from
experiments/visualize.py. Known to need debugging (commit e39091a:
"Debugging required for XVA's and greeks") — the numbers these plots show
have not been validated. Kept runnable so the debugging can happen; do not
trust the output yet.

Usage (via the visualizer):
    python experiments/visualize.py --model results/models/bs_basket_D10.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_fbsde_nn.equations import (
    BlackScholesBarenblattEquation,
    BlackScholesEquation,
)


def _get_reference_spot(eq) -> float:
    if hasattr(eq, "X0_val"):
        return float(getattr(eq, "X0_val"))
    try:
        X0 = eq.sample_initial_condition(1)
        return float(X0.mean().abs().item())
    except Exception:
        return 1.0


def compute_price_vs_spot(
    solver,
    S_min=0.5,
    S_max=1.5,
    n_points=50,
    delta_mode: str = "sum",
):
    """Compute Price and Delta vs Spot at t=0 (For BS Models)."""
    device = solver.device
    eq = solver.equation

    ref_S = _get_reference_spot(eq)
    S_vals = np.linspace(S_min * ref_S, S_max * ref_S, n_points)

    prices = []
    deltas = []

    for S in S_vals:
        # Create input (Batch=1)
        X = torch.full(
            (1, eq.D), S, device=device, dtype=torch.float32, requires_grad=True
        )

        # solver.get_price_and_delta handles eval mode and autograd internally
        u, delta = solver.get_price_and_delta(0.0, X)

        prices.append(u.item())
        if delta_mode == "mean":
            deltas.append(delta.mean().item())
        else:
            deltas.append(delta.sum().item())

    return S_vals, np.array(prices), np.array(deltas)


def _compute_bsb_exact_curves(eq, S_vals, delta_mode: str):
    device = eq.device
    prices = []
    deltas = []
    for S in S_vals:
        X = torch.full((1, eq.D), S, device=device, dtype=torch.float32)
        price = eq.exact_solution(0.0, X).item()
        prices.append(price)
        if hasattr(eq, "exact_gradient"):
            grad = eq.exact_gradient(0.0, X)
            if delta_mode == "mean":
                deltas.append(grad.mean().item())
            else:
                deltas.append(grad.sum().item())
    return np.array(prices), np.array(deltas) if deltas else None


def _compute_bs_mc_prices(eq, S_vals, mc_paths: int):
    original_spot = eq.X0_val
    prices = []
    for S in S_vals:
        eq.X0_val = float(S)
        price, _ = eq.monte_carlo_price(n_paths=mc_paths)
        prices.append(price)
    eq.X0_val = original_spot
    return np.array(prices)


def plot_bs_greeks(
    solver,
    title,
    save_dir,
    delta_mode: str,
    mc_paths: int,
    spot_min: float,
    spot_max: float,
    spot_steps: int,
):
    """Plot Price and Delta vs Spot (for BS Models)."""
    print("Computing Greeks surface...")
    S_vals, prices, deltas = compute_price_vs_spot(
        solver,
        S_min=spot_min,
        S_max=spot_max,
        n_points=spot_steps,
        delta_mode=delta_mode,
    )
    bench_prices = None
    bench_deltas = None

    if isinstance(solver.equation, BlackScholesBarenblattEquation):
        bench_prices, bench_deltas = _compute_bsb_exact_curves(
            solver.equation, S_vals, delta_mode
        )
    elif isinstance(solver.equation, BlackScholesEquation) and mc_paths > 0:
        bench_prices = _compute_bs_mc_prices(solver.equation, S_vals, mc_paths)
        bench_deltas = np.gradient(bench_prices, S_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Price
    ax1.plot(S_vals, prices, "b-", label="Deep BSDE")
    if bench_prices is not None:
        ax1.plot(S_vals, bench_prices, "k--", label="Benchmark")
    ax1.set_title("Price vs Spot (t=0)")
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Option Value")
    ax1.grid(True)
    ax1.legend()

    # Delta
    ax2.plot(S_vals, deltas, "r-", label="Delta")
    if bench_deltas is not None:
        ax2.plot(S_vals, bench_deltas, "k--", label="Benchmark")
    ax2.set_title("Delta vs Spot (t=0)")
    ax2.set_xlabel("Spot Price")
    if delta_mode == "mean":
        ax2.set_ylabel("Delta (mean ∂u/∂S)")
    else:
        ax2.set_ylabel("Delta (sum ∂u/∂S)")
    ax2.grid(True)
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "greeks.png")
    print("Saved greeks.png")
    plt.close()
