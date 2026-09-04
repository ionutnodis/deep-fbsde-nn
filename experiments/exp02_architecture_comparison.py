"""
Experiment 02: Architecture Comparison
======================================

Compare NAIS-Net vs standard FeedForward network on BSB equation.

Key hypothesis (from Güler et al. 2019):
- NAIS-Net's stability guarantees lead to more reliable training
- FeedForward may have higher variance across runs
- NAIS-Net should show smoother convergence

We run multiple trials to assess:
1. Mean and std of final error
2. Training stability (loss variance)
3. Convergence speed

Usage:
    python experiments/exp02_architecture.py
    python experiments/exp02_architecture.py --dim 100 --trials 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation
from deep_fbsde_nn.networks import FeedForwardNet, NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StandardSolver
from deep_fbsde_nn.utils import get_device


def create_network(architecture: str, input_dim: int, hidden_dim: int, num_layers: int):
    """Create network by name."""
    if architecture == "naisnet":
        return NAISNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation="sine",
        )
    else:
        return FeedForwardNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation="sine",
        )


def run_single_trial(
    architecture: str,
    dimension: int,
    n_iterations: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Run a single training trial."""

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create equation
    eq = BlackScholesBarenblattEquation(
        dimension=dimension,
        sigma_min=0.1,
        sigma_max=0.3,
        terminal_time=1.0,
        device=device,
    )

    X0 = eq.sample_initial_condition(1)
    exact_Y0 = eq.exact_solution(0.0, X0).item()

    # Create network
    input_dim = dimension + 1
    net = create_network(architecture, input_dim, hidden_dim, num_layers)

    # Solver - matching reference implementation
    config = SolverConfig(
        batch_size=1,
        num_timesteps=50,
        learning_rate=1e-3,
        num_iterations=n_iterations,
        use_mlmc=True,
        print_every=n_iterations,  # Only print at end
    )

    solver = StandardSolver(
        equation=eq,
        network=net,
        config=config,
        device=device,
    )

    # Train
    start_time = time.time()
    history = solver.train()
    train_time = time.time() - start_time

    # Evaluate
    Y0_pred = solver.predict(0.0, X0).item()
    rel_error = abs(Y0_pred - exact_Y0) / exact_Y0 * 100

    return {
        "Y0_pred": Y0_pred,
        "Y0_exact": exact_Y0,
        "relative_error_pct": rel_error,
        "train_time_sec": train_time,
        "final_loss": history["losses"][-1],
        "loss_history": history["losses"],
        "seed": seed,
    }


def run_architecture_trials(
    architecture: str,
    dimension: int,
    n_iterations: int,
    n_trials: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
    base_seed: int = 42,
) -> Dict:
    """Run multiple trials for one architecture."""

    print(f"\n{'='*60}")
    print(f"Architecture: {architecture.upper()}")
    print(f"Dimension: {dimension}, Trials: {n_trials}")
    print(f"{'='*60}")

    trials = []

    for trial in range(n_trials):
        seed = base_seed + trial
        print(f"  Trial {trial+1}/{n_trials} (seed={seed})...", end=" ", flush=True)

        result = run_single_trial(
            architecture=architecture,
            dimension=dimension,
            n_iterations=n_iterations,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
            seed=seed,
        )

        trials.append(result)
        print(f"Error: {result['relative_error_pct']:.4f}%")

    # Aggregate statistics
    errors = [t["relative_error_pct"] for t in trials]
    times = [t["train_time_sec"] for t in trials]
    losses = [t["final_loss"] for t in trials]

    return {
        "architecture": architecture,
        "dimension": dimension,
        "n_trials": n_trials,
        "n_iterations": n_iterations,
        "trials": trials,
        "stats": {
            "error_mean": np.mean(errors),
            "error_std": np.std(errors),
            "error_min": np.min(errors),
            "error_max": np.max(errors),
            "time_mean": np.mean(times),
            "loss_mean": np.mean(losses),
            "loss_std": np.std(losses),
        },
    }


def compare_architectures(
    architectures: List[str],
    dimension: int,
    n_iterations: int,
    n_trials: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
) -> List[Dict]:
    """Compare multiple architectures."""

    results = []

    for arch in architectures:
        result = run_architecture_trials(
            architecture=arch,
            dimension=dimension,
            n_iterations=n_iterations,
            n_trials=n_trials,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
        )
        results.append(result)

    return results


def print_comparison(results: List[Dict]):
    """Print comparison table."""

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'Architecture':<15} {'Error Mean':>12} {'Error Std':>12} "
        f"{'Loss Mean':>12} {'Time (s)':>10}"
    )
    print(f"{'-'*70}")

    for r in results:
        s = r["stats"]
        print(
            f"{r['architecture']:<15} {s['error_mean']:>12.4f} {s['error_std']:>12.4f} "
            f"{s['loss_mean']:>12.2e} {s['time_mean']:>10.1f}"
        )

    print(f"{'-'*70}")

    # Statistical comparison
    if len(results) == 2:
        e1 = [t["relative_error_pct"] for t in results[0]["trials"]]
        e2 = [t["relative_error_pct"] for t in results[1]["trials"]]

        # Simple comparison
        better = (
            results[0]["architecture"]
            if np.mean(e1) < np.mean(e2)
            else results[1]["architecture"]
        )
        improvement = (
            abs(np.mean(e1) - np.mean(e2)) / max(np.mean(e1), np.mean(e2)) * 100
        )

        print(f"\n{better} has {improvement:.1f}% lower mean error")


def save_results(results: List[Dict], output_dir: str = "results"):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"exp02_architecture_{timestamp}.json"

    # Clean for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = []
    for r in results:
        r_clean = {}
        for k, v in r.items():
            if isinstance(v, dict):
                r_clean[k] = {kk: convert(vv) for kk, vv in v.items()}
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                r_clean[k] = [{kk: convert(vv) for kk, vv in d.items()} for d in v]
            else:
                r_clean[k] = convert(v)
        results_clean.append(r_clean)

    with open(filename, "w") as f:
        json.dump(
            {
                "experiment": "exp02_architecture",
                "timestamp": timestamp,
                "results": results_clean,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Architecture Comparison Experiment")
    parser.add_argument("--dim", type=int, default=100, help="Dimension")
    parser.add_argument(
        "--iterations", type=int, default=5000, help="Training iterations per trial"
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of trials per architecture"
    )
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    # Run comparison
    results = compare_architectures(
        architectures=["naisnet", "feedforward"],
        dimension=args.dim,
        n_iterations=args.iterations,
        n_trials=args.trials,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        device=device,
    )

    # Print and save
    print_comparison(results)

    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()
