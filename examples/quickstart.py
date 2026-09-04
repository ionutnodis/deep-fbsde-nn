"""
Quickstart: solve the Black-Scholes-Barenblatt equation with Deep BSDE.

This is the README example as a runnable, parameterized script; CI executes
it at reduced scale (--dim 4 --iterations 200) so the documentation cannot
silently rot.

Usage:
    python examples/quickstart.py                        # laptop scale
    python examples/quickstart.py --dim 100 --iterations 5000   # paper scale
"""

import argparse

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StandardSolver
from deep_fbsde_nn.utils import get_device


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=10, help="problem dimension D")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)

    equation = BlackScholesBarenblattEquation(
        dimension=args.dim, sigma_min=0.1, sigma_max=0.3,
        terminal_time=1.0, device=device,
    )
    network = NAISNet(
        input_dim=args.dim + 1, hidden_dim=args.hidden,
        output_dim=1, num_layers=4, activation="sine",
    )
    config = SolverConfig(
        batch_size=args.batch, num_timesteps=args.timesteps,
        learning_rate=1e-3, num_iterations=args.iterations, use_mlmc=False,
        print_every=max(args.iterations // 5, 1),
    )
    solver = StandardSolver(equation, network, config, device=device)
    solver.train()

    results = solver.validate()
    print(f"Relative error vs exact solution: {results['relative_error']:.2f}%")
    return results


if __name__ == "__main__":
    main()
