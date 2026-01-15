"""
Experiment 01: BSB Dimension Scaling
====================================

Test Deep BSDE on Black-Scholes-Barenblatt equation across dimensions.

This is the core experiment demonstrating that the method scales to
high-dimensional PDEs (D = 10, 50, 100, 200).

The BSB equation has an analytical solution, allowing precise error measurement:
    u(t, x) = ||x||² · exp(σ_max² · (T - t))

Expected results (from literature):
    D=10:  ~0.1-0.5% error
    D=50:  ~0.3-1.0% error
    D=100: ~0.5-2.0% error
    D=200: ~1.0-3.0% error

Usage:
    python experiments/exp01_bsb_dimension.py
    python experiments/exp01_bsb_dimension.py --dim 100 --iterations 5000
    python experiments/exp01_bsb_dimension.py --all
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import time
import json
from datetime import datetime

from deep_fbsde_nn.networks import NAISNet, FeedForwardNet
from deep_fbsde_nn.equations import BlackScholesBarenblattEquation
from deep_fbsde_nn.solvers import StandardSolver, SolverConfig
from deep_fbsde_nn.utils import get_device
from experiments.config import BSB_CONFIG


def create_network(architecture: str, input_dim: int, hidden_dim: int, num_layers: int):
    """Create network based on architecture name."""
    if architecture == "naisnet":
        return NAISNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation="sine",
        )
    elif architecture == "feedforward":
        return FeedForwardNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation="sine",
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def run_single_dimension(
    dimension: int,
    config,
    device: torch.device,
) -> dict:
    """
    Run experiment for a single dimension.

    Returns:
        Dict with results
    """
    print(f"\n{'='*60}")
    print(f"BSB Experiment: D = {dimension}")
    print(f"{'='*60}")

    # Create equation
    eq = BlackScholesBarenblattEquation(
        dimension=dimension,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        terminal_time=config.terminal_time,
        device=device,
    )

    # Initial condition and exact solution
    X0 = eq.sample_initial_condition(1)
    exact_Y0 = eq.exact_solution(0.0, X0).item()

    print(f"Initial condition: x = (1/√{dimension}, ..., 1/√{dimension})")
    print(f"||X0||² = {(X0**2).sum().item():.4f}")
    print(f"Exact Y0 = {exact_Y0:.6f}")

    # Create network
    input_dim = dimension + 1  # (t, X)
    net = create_network(
        config.architecture,
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    print(f"Network: {config.architecture}, {net.count_parameters():,} parameters")

    # Solver config - matching reference implementation
    solver_config = SolverConfig(
        batch_size=1,
        num_timesteps=50,
        learning_rate=1e-3,
        num_iterations=config.n_iterations,
        use_mlmc=True,
        print_every=100,
    )

    # Create solver
    solver = StandardSolver(
        equation=eq,
        network=net,
        config=solver_config,
        device=device,
    )

    # Train
    print(f"\nTraining for {config.n_iterations} iterations...")
    start_time = time.time()
    history = solver.train()
    train_time = time.time() - start_time

    # Save model
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"bsb_D{dimension}.pt"
    solver.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Validate
    Y0_pred = solver.predict(0.0, X0).item()
    rel_error = abs(Y0_pred - exact_Y0) / exact_Y0 * 100

    print(f"\n{'='*60}")
    print(f"Results for D = {dimension}")
    print(f"{'='*60}")
    print(f"Y0 predicted:   {Y0_pred:.6f}")
    print(f"Y0 exact:       {exact_Y0:.6f}")
    print(f"Relative error: {rel_error:.4f}%")
    print(f"Training time:  {train_time:.1f}s")
    print(f"Final loss:     {history['losses'][-1]:.3e}")

    return {
        "dimension": dimension,
        "Y0_pred": Y0_pred,
        "Y0_exact": exact_Y0,
        "relative_error_pct": rel_error,
        "train_time_sec": train_time,
        "final_loss": history["losses"][-1],
        "iterations": config.n_iterations,
        "architecture": config.architecture,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "model_path": str(model_path),
        "history": {
            "iterations": history["iterations"],
            "losses": history["losses"],
            "Y0": history["Y0"],
        },
    }


def run_all_dimensions(config, device: torch.device) -> list:
    """Run experiment for all dimensions in config."""
    results = []

    for dim in config.dimensions:
        result = run_single_dimension(dim, config, device)
        results.append(result)

    return results


def print_summary(results: list):
    """Print summary table of results."""
    print(f"\n{'='*70}")
    print("SUMMARY: BSB Dimension Scaling")
    print(f"{'='*70}")
    print(
        f"{'Dim':>6} {'Y0_pred':>12} {'Y0_exact':>12} {'Error %':>10} {'Time (s)':>10}"
    )
    print(f"{'-'*70}")

    for r in results:
        print(
            f"{r['dimension']:>6} {r['Y0_pred']:>12.6f} {r['Y0_exact']:>12.6f} "
            f"{r['relative_error_pct']:>10.4f} {r['train_time_sec']:>10.1f}"
        )

    print(f"{'-'*70}")

    # Summary statistics
    errors = [r["relative_error_pct"] for r in results]
    print(f"Mean error: {np.mean(errors):.4f}%")
    print(f"Max error:  {np.max(errors):.4f}%")


def save_results(results: list, output_dir: str = "results"):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"exp01_bsb_dimension_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = []
    for r in results:
        r_clean = {
            k: convert(v) if not isinstance(v, dict) else v for k, v in r.items()
        }
        results_clean.append(r_clean)

    with open(filename, "w") as f:
        json.dump(
            {
                "experiment": "exp01_bsb_dimension",
                "timestamp": timestamp,
                "results": results_clean,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="BSB Dimension Scaling Experiment")
    parser.add_argument("--dim", type=int, default=100, help="Single dimension to test")
    parser.add_argument(
        "--all", action="store_true", help="Run all dimensions [10, 50, 100, 200]"
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Override number of iterations"
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--arch",
        type=str,
        default="naisnet",
        choices=["naisnet", "feedforward"],
        help="Network architecture",
    )
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu"
    )
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create config with overrides
    config = BSB_CONFIG
    if args.iterations:
        config.n_iterations = args.iterations
    if args.batch:
        config.batch_size = args.batch
    config.architecture = args.arch
    config.hidden_dim = args.hidden
    config.num_layers = args.layers

    # Run experiment(s)
    if args.all:
        results = run_all_dimensions(config, device)
    else:
        config.dimensions = [args.dim]
        results = [run_single_dimension(args.dim, config, device)]

    # Print summary
    print_summary(results)

    # Save results
    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()
