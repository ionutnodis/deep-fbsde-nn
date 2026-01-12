"""
Experiment 03: Black-Scholes Basket Options
============================================

Price high-dimensional basket options using Deep BSDE.

Unlike BSB, Black-Scholes basket options don't have closed-form solutions
(except geometric baskets). We validate against Monte Carlo benchmarks.

This experiment demonstrates:
1. Pricing arithmetic basket calls in D = 10, 50, 100 dimensions
2. Comparison with Monte Carlo (ground truth)
3. Delta hedging accuracy

Option setup:
- Arithmetic basket call: max(mean(S_T) - K, 0)
- S_0 = K = 100 (ATM)
- r = 5%, σ = 20%, T = 1 year
- Uncorrelated assets (ρ = 0)

Usage:
    python experiments/exp03_black_scholes.py
    python experiments/exp03_black_scholes.py --dim 100 --iterations 10000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import time
import json
from datetime import datetime

from src.networks import NAISNet
from src.equations import BlackScholesEquation
from src.solvers import StandardSolver, SolverConfig
from src.utils import get_device
from experiments.config import BS_CONFIG


def run_single_dimension(
    dimension: int,
    n_iterations: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    device: torch.device = None,
    mc_paths: int = 500000,
    save_model: bool = True,
    output_dir: str = "results/models",
) -> dict:
    """
    Run Black-Scholes experiment for a single dimension.
    """

    print(f"\n{'='*60}")
    print(f"Black-Scholes Basket Call: D = {dimension}")
    print(f"{'='*60}")

    # Create equation - uses reference defaults (r=0.01, σ=0.25, strike=D, X0=1.0)
    eq = BlackScholesEquation(
        dimension=dimension,
        device=device,
    )

    print(f"Parameters: r={eq.r}, σ={eq._sigma_val}, T={eq.T}")
    print(f"Strike = {eq.strike} (= D), X0 = {eq.X0_val} per asset")

    # Monte Carlo benchmark
    print(f"\nComputing Monte Carlo benchmark ({mc_paths:,} paths)...")
    mc_start = time.time()
    mc_price, mc_stderr = eq.monte_carlo_price(n_paths=mc_paths)
    mc_time = time.time() - mc_start
    print(f"MC Price: {mc_price:.6f} ± {mc_stderr:.6f} ({mc_time:.1f}s)")

    # Create network
    input_dim = dimension + 1
    net = NAISNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        activation="sine",
    )
    print(f"Network: NAISNet, {net.count_parameters():,} parameters")

    # Solver configuration
    config = SolverConfig(
        batch_size=128,  # Reference uses M=1
        num_timesteps=50,
        learning_rate=3e-3,  # Reference uses 1e-3
        num_iterations=n_iterations,
        use_mlmc=True,
        mlmc_schedule={0: 5, 1500: 10, 3000: 25, 5000: 50},
        print_every=500,
        gradient_clip=1.0,  # Reference uses 1.0
    )

    solver = StandardSolver(
        equation=eq,
        network=net,
        config=config,
        device=device,
    )

    # Train
    print(f"\nTraining for {n_iterations} iterations...")
    start_time = time.time()
    history = solver.train()
    train_time = time.time() - start_time

    # Save model
    if save_model:
        model_dir = Path(output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"bs_basket_D{dimension}.pt"
        solver.save(str(model_path))
        print(f"Model saved to {model_path}")

    # Evaluate at S_0 = K (ATM)
    X0 = eq.sample_initial_condition(1)  # All assets at strike
    Y0_pred = solver.predict(0.0, X0).item()

    # Error vs Monte Carlo
    abs_error = abs(Y0_pred - mc_price)
    rel_error = abs_error / mc_price * 100

    # Check if within MC confidence interval
    within_ci = abs_error < 2 * mc_stderr

    print(f"\n{'='*60}")
    print(f"Results for D = {dimension}")
    print(f"{'='*60}")
    print(f"Deep BSDE price: {Y0_pred:.6f}")
    print(f"MC benchmark:    {mc_price:.6f} ± {mc_stderr:.6f}")
    print(f"Absolute error:  {abs_error:.6f}")
    print(f"Relative error:  {rel_error:.4f}%")
    print(f"Within 2σ CI:    {'Yes' if within_ci else 'No'}")
    print(f"Training time:   {train_time:.1f}s")

    return {
        "dimension": dimension,
        "strike": eq.strike,
        "r": eq.r,
        "sigma": eq._sigma_val,
        "T": eq.T,
        "Y0_pred": Y0_pred,
        "mc_price": mc_price,
        "mc_stderr": mc_stderr,
        "abs_error": abs_error,
        "rel_error_pct": rel_error,
        "within_ci": within_ci,
        "train_time_sec": train_time,
        "mc_time_sec": mc_time,
        "final_loss": history["losses"][-1],
        "n_iterations": n_iterations,
        "model_path": str(model_path) if save_model else None,
    }


def run_all_dimensions(config, device: torch.device, save_model: bool = True) -> list:
    """Run for all dimensions in config."""
    results = []

    for dim in config.dimensions:
        result = run_single_dimension(
            dimension=dim,
            n_iterations=config.n_iterations,
            device=device,
            save_model=save_model,
        )
        results.append(result)

    return results


def print_summary(results: list):
    """Print summary table."""

    print(f"\n{'='*80}")
    print("SUMMARY: Black-Scholes Basket Options")
    print(f"{'='*80}")
    print(
        f"{'Dim':>6} {'BSDE Price':>12} {'MC Price':>12} {'MC Stderr':>10} "
        f"{'Error %':>10} {'In CI':>8}"
    )
    print(f"{'-'*80}")

    for r in results:
        ci_str = "Yes" if r["within_ci"] else "No"
        print(
            f"{r['dimension']:>6} {r['Y0_pred']:>12.6f} {r['mc_price']:>12.6f} "
            f"{r['mc_stderr']:>10.6f} {r['rel_error_pct']:>10.4f} {ci_str:>8}"
        )

    print(f"{'-'*80}")

    # Summary
    all_in_ci = all(r["within_ci"] for r in results)
    mean_error = np.mean([r["rel_error_pct"] for r in results])
    print(f"Mean relative error: {mean_error:.4f}%")
    print(f"All within MC confidence interval: {'Yes' if all_in_ci else 'No'}")


def save_results(results: list, output_dir: str = "results"):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"exp03_black_scholes_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return obj
        return obj

    results_clean = [{k: convert(v) for k, v in r.items()} for r in results]

    with open(filename, "w") as f:
        json.dump(
            {
                "experiment": "exp03_black_scholes",
                "timestamp": timestamp,
                "results": results_clean,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Black-Scholes Basket Option Experiment"
    )
    parser.add_argument(
        "--dim", type=int, default=None, help="Single dimension (default: run all)"
    )
    parser.add_argument(
        "--iterations", type=int, default=20000, help="Training iterations"
    )
    parser.add_argument(
        "--mc-paths", type=int, default=500000, help="Monte Carlo paths for benchmark"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    # Update config
    config = BS_CONFIG
    config.n_iterations = args.iterations

    # Run
    if args.dim is not None:
        results = [
            run_single_dimension(
                dimension=args.dim,
                n_iterations=args.iterations,
                device=device,
                mc_paths=args.mc_paths,
                save_model=True,  # Always save
            )
        ]
    else:
        results = run_all_dimensions(config, device, save_model=True)  # Always save

    # Summary and save
    print_summary(results)

    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()
