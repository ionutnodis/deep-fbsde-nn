"""
Reproducible benchmark: every equation with a reference, across dimensions.

One command regenerates the table the README publishes:

    python benchmarks/run.py            # full matrix (~10 min CPU)
    python benchmarks/run.py --quick    # low-dim subset (~2 min, sanity)

Outputs:
    benchmarks/results.json   raw numbers (append-safe, one run per file)
    benchmarks/results.md     the rendered table
    README.md                 table spliced between BENCHMARK markers

Every run is seeded; references are exact solutions, published benchmark
values, or seeded Monte-Carlo (noted per row). Timings are wall-clock on the
machine that ran it — regenerate on your own hardware rather than trusting
ours.
"""

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from deep_fbsde_nn.equations import (
    AllenCahnEquation,
    BlackScholesBarenblattEquation,
    BlackScholesEquation,
    HJBEquation,
    VanillaCallEquation,
)
from deep_fbsde_nn.networks import NAISNet
from deep_fbsde_nn.solvers import SolverConfig, StandardSolver, StepwiseSolver

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS_MD = Path(__file__).parent / "results.md"
RESULTS_JSON = Path(__file__).parent / "results.json"
START_MARK = "<!-- BENCHMARK:START -->"
END_MARK = "<!-- BENCHMARK:END -->"


@dataclass
class Case:
    equation: str
    dimension: int
    solver: str
    reference_kind: str
    run: Callable[[], tuple]  # -> (predicted, reference, seconds)
    quick: bool = False  # included in --quick subset


def _stepwise(eq, batch, lr, iters, hidden, fine_lr=None, fine_iters=0, timesteps=20):
    config = SolverConfig(
        batch_size=batch, num_timesteps=timesteps, learning_rate=lr,
        num_iterations=iters, use_mlmc=False, print_every=10**9,
    )
    solver = StepwiseSolver(eq, config, device="cpu", hidden_dim=hidden)
    t0 = time.perf_counter()
    solver.train(n_iter=iters)
    if fine_iters:
        for group in solver.optimizer.param_groups:
            group["lr"] = fine_lr
        solver.train(n_iter=fine_iters)
    return solver.predict().item(), time.perf_counter() - t0


def _standard(eq, batch, lr, iters, hidden, fine_lr=None, fine_iters=0, timesteps=16):
    net = NAISNet(input_dim=eq.D + 1, hidden_dim=hidden, output_dim=1, num_layers=3)
    config = SolverConfig(
        batch_size=batch, num_timesteps=timesteps, learning_rate=lr,
        num_iterations=iters, use_mlmc=False, print_every=10**9,
    )
    solver = StandardSolver(eq, net, config, device="cpu")
    t0 = time.perf_counter()
    solver.train(n_iter=iters)
    if fine_iters:
        for group in solver.optimizer.param_groups:
            group["lr"] = fine_lr
        solver.train(n_iter=fine_iters)
    y0 = solver.predict(0.0, eq.sample_initial_condition(1)).item()
    return y0, time.perf_counter() - t0


def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def bsb_stepwise(dim, quick=False):
    def run():
        _seed()
        eq = BlackScholesBarenblattEquation(dimension=dim)
        exact = eq.exact_solution(0.0, eq.sample_initial_condition(1)).item()
        hidden = max(32, dim + 10)
        pred, secs = _stepwise(eq, batch=128, lr=1e-3, iters=2000, hidden=hidden)
        return pred, exact, secs

    return Case("Black-Scholes-Barenblatt", dim, "Stepwise", "exact solution", run, quick)


def bsb_standard():
    def run():
        _seed()
        eq = BlackScholesBarenblattEquation(dimension=2)
        exact = eq.exact_solution(0.0, eq.sample_initial_condition(1)).item()
        pred, secs = _standard(eq, batch=16, lr=5e-3, iters=600, hidden=32)
        return pred, exact, secs

    return Case("Black-Scholes-Barenblatt", 2, "Standard", "exact solution", run, quick=True)


def vanilla_standard():
    def run():
        _seed()
        eq = VanillaCallEquation(dimension=1)
        exact = eq.exact_solution(0.0, torch.tensor([[1.0]])).item()
        pred, secs = _standard(
            eq, batch=16, lr=5e-3, iters=1500, hidden=32, fine_lr=1e-3, fine_iters=700
        )
        return pred, exact, secs

    return Case("Vanilla call", 1, "Standard", "Black-Scholes closed form", run, quick=True)


def hjb_stepwise(dim, quick=False):
    def run():
        _seed()
        eq = HJBEquation(dimension=dim)
        reference = eq.exact_solution(
            0.0, eq.sample_initial_condition(1), n_mc=400_000,
            generator=torch.Generator().manual_seed(0),
        ).item()
        if dim <= 10:
            pred, secs = _stepwise(
                eq, batch=64, lr=5e-3, iters=2000, hidden=32, fine_lr=1e-3, fine_iters=1000
            )
        else:
            pred, secs = _stepwise(
                eq, batch=256, lr=5e-4, iters=3000, hidden=dim + 10
            )
        return pred, reference, secs

    kind = "Cole-Hopf MC (seeded)" + (" / published 4.5901" if dim == 100 else "")
    return Case("Hamilton-Jacobi-Bellman", dim, "Stepwise", kind, run, quick)


def allen_cahn_stepwise():
    def run():
        _seed()
        eq = AllenCahnEquation(dimension=100)
        pred, secs = _stepwise(eq, batch=256, lr=5e-4, iters=3000, hidden=110)
        return pred, AllenCahnEquation.KNOWN_VALUE_D100, secs

    return Case(
        "Allen-Cahn", 100, "Stepwise", "published 0.052802 (branching diffusion)", run
    )


def basket_stepwise(dim, quick=False):
    def run():
        _seed()
        eq = BlackScholesEquation(dimension=dim)
        mc, _se = eq.monte_carlo_price(n_paths=500_000)
        pred, secs = _stepwise(
            eq, batch=64, lr=5e-3, iters=2000, hidden=max(32, dim + 10),
            fine_lr=1e-3, fine_iters=1000,
        )
        return pred, mc, secs

    return Case("BS basket call", dim, "Stepwise", "seeded MC (same payoff)", run, quick)


CASES = [
    bsb_standard(),
    bsb_stepwise(10, quick=True),
    bsb_stepwise(50),
    bsb_stepwise(100),
    vanilla_standard(),
    basket_stepwise(5, quick=True),
    basket_stepwise(25),
    hjb_stepwise(3, quick=True),
    hjb_stepwise(20),
    hjb_stepwise(100),
    allen_cahn_stepwise(),
]


def render_table(rows: list) -> str:
    lines = [
        "| Equation | d | Solver | Reference | Rel. error | Time |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['equation']} | {r['dimension']} | {r['solver']} | "
            f"{r['reference_kind']} | {r['rel_error_pct']:.2f}% | {r['seconds']:.0f}s |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="low-dim subset only")
    parser.add_argument(
        "--no-readme", action="store_true", help="don't splice the table into README"
    )
    args = parser.parse_args()

    cases = [c for c in CASES if c.quick] if args.quick else CASES
    rows = []
    for case in cases:
        print(f"[{case.equation} d={case.dimension} {case.solver}] running...", flush=True)
        pred, ref, secs = case.run()
        rel = abs(pred - ref) / abs(ref) * 100
        rows.append(
            {
                "equation": case.equation,
                "dimension": case.dimension,
                "solver": case.solver,
                "reference_kind": case.reference_kind,
                "predicted": pred,
                "reference": ref,
                "rel_error_pct": rel,
                "seconds": secs,
            }
        )
        print(f"  pred={pred:.6f} ref={ref:.6f} err={rel:.2f}% ({secs:.0f}s)", flush=True)

    table = render_table(rows)
    meta = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "quick": args.quick,
    }
    RESULTS_JSON.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    RESULTS_MD.write_text(
        f"# Benchmark results\n\ntorch {meta['torch']} · {meta['platform']} · CPU\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {RESULTS_JSON.name}, {RESULTS_MD.name}")

    if not args.no_readme and not args.quick:
        readme = README.read_text()
        if START_MARK in readme and END_MARK in readme:
            head, rest = readme.split(START_MARK, 1)
            _, tail = rest.split(END_MARK, 1)
            block = (
                f"{START_MARK}\n{table}\n\n"
                f"<sub>Seeded runs, CPU ({meta['processor']}), torch {meta['torch']}. "
                f"Regenerate: `python benchmarks/run.py` (~10 min).</sub>\n{END_MARK}"
            )
            README.write_text(head + block + tail)
            print("README benchmark section updated")
        else:
            print("README markers not found — table not spliced")


if __name__ == "__main__":
    main()
