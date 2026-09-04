"""
Experiment: Hamilton-Jacobi-Bellman (HJB)
=========================================

Solves the high-dimensional HJB equation for stochastic optimal control.
Standard benchmark problem from Han et al. (2018).

Problem:
    ∂u/∂t + Δu - λ||∇u||² = 0
    D = 100
    X0 = 0

This problem tests the solver's ability to handle:
1. High dimensions (D=100)
2. Quadratic gradient nonlinearity in the driver
3. Control-type value functions

Usage:
    python experiments/exp_hjb.py --dim 100 --iterations 5000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_fbsde_nn.equations.hjb import HJBEquation
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StandardSolver
from deep_fbsde_nn.utils import get_device


def run_hjb_experiment(
    dimension: int = 100,
    n_iterations: int = 5000,
    batch_size: int = 64,
    hidden_dim: int = 256,
    num_layers: int = 4,
    lambda_val: float = 1.0,
    device: str = "auto",
    save_dir: str = "results/hjb",
):
    # Setup
    device = get_device(device)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"HJB Equation Experiment (D={dimension})")
    print(f"{'='*60}")

    # 1. Equation
    eq = HJBEquation(dimension=dimension, lambda_val=lambda_val, device=device)

    # 2. Compute Benchmark (Monte Carlo)
    print("\nComputing Monte Carlo benchmark (this may take a moment)...")
    mc_start = time.time()
    # Using 200k paths for high precision reference
    ref_u = eq.exact_solution(0.0, n_mc=200000).item()
    mc_time = time.time() - mc_start
    print(f"Benchmark Price u(0,0): {ref_u:.6f} (Time: {mc_time:.2f}s)")

    # 3. Network (NAIS-Net)
    # Using SiLU activation as per our robust findings
    net = NAISNet(
        input_dim=dimension + 1,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        activation="silu",
    ).to(device)

    print(f"Network: NAISNet ({sum(p.numel() for p in net.parameters()):,} params)")

    # 4. Solver Config
    config = SolverConfig(
        batch_size=batch_size,
        num_timesteps=50,  # Standard for HJB
        learning_rate=1e-3,  # Standard LR for HJB
        num_iterations=n_iterations,
        use_mlmc=True,  # Use MLMC for speedup
        mlmc_schedule={0: 10, 1000: 20, 3000: 50},
        print_every=100,
    )

    solver = StandardSolver(eq, net, config, device=device)

    # 5. Training
    print(f"\nStarting training ({n_iterations} iterations)...")
    history = solver.train()

    # 6. Evaluation
    final_pred = solver.predict(0.0, eq.sample_initial_condition(1)).item()
    rel_error = abs(final_pred - ref_u) / abs(ref_u) * 100

    # Save model
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"hjb_D{dimension}.pt"
    solver.save(str(model_path))
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"True Value:     {ref_u:.6f}")
    print(f"Deep BSDE:      {final_pred:.6f}")
    print(f"Relative Error: {rel_error:.4f}%")
    print(f"{'='*60}")

    # 7. Plots
    plot_hjb_results(history, ref_u, save_path / f"hjb_d{dimension}.png")

    # Save Data
    results = {
        "dimension": dimension,
        "true_value": ref_u,
        "pred_value": final_pred,
        "relative_error": rel_error,
        "config": str(config),
        "loss_history": history["losses"],
        "y0_history": history["Y0"],
    }

    with open(save_path / f"hjb_d{dimension}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


def plot_hjb_results(history, true_val, save_path):
    """Plot Loss and Convergence."""
    losses = history["losses"]
    y0_hist = history["Y0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Loss
    ax1.semilogy(losses)
    ax1.set_title("Loss Convergence (Log Scale)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which="both", alpha=0.3)

    # Solution Value
    ax2.plot(y0_hist, label="Deep BSDE")
    ax2.axhline(y=true_val, color="r", linestyle="--", label=f"Exact: {true_val:.4f}")
    ax2.set_title("Y(0) Convergence")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("u(0, x=0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plots saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_hjb_experiment(
        dimension=args.dim, n_iterations=args.iterations, device=args.device
    )
