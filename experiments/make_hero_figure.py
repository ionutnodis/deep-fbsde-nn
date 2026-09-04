"""
Regenerate the README hero figure: a seeded D=100 BSB training run showing
predicted vs exact solution values along sample paths.

Release evidence, not decoration: this run backs the README's
high-dimension claim, so it is regenerated (not reused) before each release,
AFTER any change to network math (release checklist in CONTRIBUTING.md).

Usage:
    python experiments/make_hero_figure.py                # full D=100 run
    python experiments/make_hero_figure.py --fast         # tiny CI smoke

Requires the [experiments] extra (matplotlib).
Output: docs/assets/hero_bsb_d100.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_fbsde_nn.equations import BlackScholesBarenblattEquation
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StandardSolver
from deep_fbsde_nn.utils import get_device

OUT = Path(__file__).parent.parent / "docs" / "assets" / "hero_bsb_d100.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="tiny smoke run for CI")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    dim = 4 if args.fast else 100
    iters = 100 if args.fast else 4000
    hidden = 32 if args.fast else 256

    device = get_device(args.device)
    eq = BlackScholesBarenblattEquation(dimension=dim, device=device)
    net = NAISNet(input_dim=dim + 1, hidden_dim=hidden, output_dim=1, num_layers=4)
    config = SolverConfig(
        batch_size=16, num_timesteps=20, learning_rate=1e-3,
        num_iterations=iters, use_mlmc=False, print_every=max(iters // 8, 1),
    )
    solver = StandardSolver(eq, net, config, device=device)
    solver.train()
    result = solver.validate()

    # Predicted vs exact along a few sample paths
    n_paths, n_steps = 6, 40
    t, dW, _ = solver.generate_paths(n_paths, n_steps)
    X = eq.sample_initial_condition(n_paths)
    times = t.cpu().numpy()
    pred, exact = [], []
    with torch.no_grad():
        for n in range(n_steps + 1):
            tn = float(times[n])
            pred.append(solver.predict(tn, X).cpu().numpy().ravel())
            exact.append(eq.exact_solution(tn, X).cpu().numpy().ravel())
            if n < n_steps:
                sigma = eq.diffusion(t[n], X, None)
                X = X + sigma * dW[:, n, :]

    pred = np.array(pred)
    exact = np.array(exact)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for i in range(n_paths):
        ax.plot(times, exact[:, i], "k--", alpha=0.5, lw=1)
        ax.plot(times, pred[:, i], alpha=0.85, lw=1.4)
    ax.plot([], [], "k--", label="exact $u(t, X_t)$")
    ax.plot([], [], "-", color="tab:blue", label="Deep BSDE prediction")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("$u(t, X_t)$")
    ax.set_title(
        f"Black-Scholes-Barenblatt, $d={dim}$ — "
        f"rel. error at $t{{=}}0$: {result['relative_error']:.2f}%"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT} (relative error {result['relative_error']:.2f}%)")


if __name__ == "__main__":
    main()
