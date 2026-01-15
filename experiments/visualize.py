"""
Visualization Script
====================

Universal visualizer for Deep BSDE models (BS, BSB, HJB, XVA).
Automatically detects equation type and dimension from filename.

Usage:
    python experiments/visualize.py --model results/models/bs_basket_D10.pt
    python experiments/visualize.py --model results/models/hjb_D100.pt
"""

import sys
import json
from pathlib import Path
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports
from src.networks import NAISNet
from src.networks.wrapper import BlackScholesWrapper
from src.equations import (
    BlackScholesBarenblattEquation,
    BlackScholesEquation,
    HJBEquation,
    BaseEquation,
    EquationConfig,
)
from src.solvers import StandardSolver, SolverConfig
from src.utils import get_device


# --- MOCK CLASSES FOR UNSAFE PICKLE LOADING ---
@dataclass
class XVAParams:
    """Mock class to allow unpickling XVA models."""

    # Underlying
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2

    # Credit parameters
    lambda_c: float = 0.02
    lambda_b: float = 0.01
    R_c: float = 0.4
    R_b: float = 0.4

    # Funding parameters
    r_f: float = 0.06

    # Option type
    option_type: str = "call"


# Inject into main module scope so torch.load can find it
if "__main__" in sys.modules:
    sys.modules["__main__"].XVAParams = XVAParams

# Plot settings
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# --- HELPER CLASSES ---
class XVAEquation(BaseEquation):
    """Placeholder if XVA class is missing."""

    def __init__(self, dimension, terminal_time=1.0, device=None):
        super().__init__(EquationConfig("XVA", dimension, terminal_time), device)

    def drift(self, t, x, y, z):
        return torch.zeros_like(x)

    def diffusion(self, t, x, y):
        return torch.full_like(x, 0.2)

    def driver(self, t, x, y, z):
        return torch.zeros_like(y)

    def terminal(self, x):
        return torch.zeros((x.shape[0], 1), device=self.device)


# --- FACTORY ---
def get_equation(eq_type: str, dim: int, device):
    eq_type = eq_type.lower()
    if "hjb" in eq_type:
        return HJBEquation(dimension=dim, device=device)
    elif "bsb" in eq_type:
        return BlackScholesBarenblattEquation(dimension=dim, device=device)
    elif "bs" in eq_type:
        return BlackScholesEquation(dimension=dim, device=device)
    elif "xva" in eq_type:
        return XVAEquation(dimension=dim, device=device)
    else:
        print(f"Warning: Unknown equation '{eq_type}', defaulting to BSB.")
        return BlackScholesBarenblattEquation(dimension=dim, device=device)


# --- DATA GENERATION ---
def generate_paths_and_values(
    solver: StandardSolver, n_paths: int = 100, n_steps: int = 50
) -> Dict[str, np.ndarray]:
    """Generate sample paths and network predictions."""
    eq = solver.equation
    device = solver.device
    D = eq.D
    T = eq.T
    dt = T / n_steps

    t = torch.linspace(0, T, n_steps + 1, device=device)

    try:
        X0 = eq.sample_initial_condition(n_paths)
    except:
        X0 = torch.zeros(n_paths, D, device=device)

    Y_learned = torch.zeros(n_paths, n_steps + 1, device=device)
    Y_exact = None

    try:
        if eq.has_exact_solution():
            Y_exact = torch.full((n_paths, n_steps + 1), float("nan"), device=device)
    except Exception:
        pass

    X = X0.clone()

    # We loop through time
    for n in range(n_steps + 1):
        t_n = t[n].item()

        # 1. Forward Pass (Value & Gradient)
        t_batch = torch.full((n_paths, 1), t_n, device=device)

        # IMPORTANT: Enable grad for Z computation, but detach X to save memory graph
        X_curr = X.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            Y_pred, _ = solver.net_u(t_batch, X_curr)

        Y_learned[:, n] = Y_pred.detach().squeeze()

        # 2. Exact Solution
        if Y_exact is not None:
            # Skip heavy MC for HJB intermediate steps
            if "Hamilton" in eq.config.name:
                if n == 0 or n == n_steps:
                    try:
                        Y_ex = eq.exact_solution(t_n, X_curr, n_mc=1000)
                        Y_exact[:, n] = Y_ex.detach().squeeze()
                    except:
                        pass
            else:
                try:
                    Y_ex = eq.exact_solution(t_n, X_curr)
                    Y_exact[:, n] = Y_ex.detach().squeeze()
                except:
                    pass

        # 3. Evolve SDE (Euler-Maruyama)
        if n < n_steps:
            with torch.no_grad():
                dW = torch.randn(n_paths, D, device=device) * np.sqrt(dt)

                if (
                    hasattr(eq, "L")
                    and hasattr(eq, "has_correlation")
                    and eq.has_correlation
                ):
                    dW = dW @ eq.L.t()

                Y_val = Y_pred.detach()
                mu = eq.drift(t_n, X, Y_val, None)
                sigma = eq.diffusion(t_n, X, Y_val)

                X = X + mu * dt + sigma * dW

    terminal_payoff = None
    try:
        terminal_payoff = eq.terminal(X).detach().squeeze()
    except Exception:
        pass

    return {
        "t": t.cpu().numpy(),
        "Y_learned": Y_learned.cpu().numpy(),
        "Y_exact": Y_exact.cpu().numpy() if Y_exact is not None else None,
        "terminal_payoff": (
            terminal_payoff.cpu().numpy() if terminal_payoff is not None else None
        ),
    }


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


# --- PLOTTING FUNCTIONS ---
def _get_iterations(history: Dict) -> np.ndarray:
    iterations = history.get("iterations")
    if iterations:
        return np.array(iterations)
    losses = history.get("losses", [])
    return np.arange(len(losses))


def _to_numpy(values):
    if isinstance(values, np.ndarray):
        return values
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def plot_learning_history(history, title, save_path, exact_y0: Optional[float] = None):
    """Plot Loss and Y0 Convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    iterations = _get_iterations(history)

    # Loss
    if "losses" in history:
        ax1.semilogy(iterations, history["losses"])
        ax1.set_title("Loss Convergence")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.grid(True, which="both", alpha=0.3)

    # Y0
    # Handle both key names just in case
    y0_data = history.get("Y0") or history.get("Y0_history")
    if y0_data:
        ax2.plot(iterations, y0_data, label="Learned Y0")
        ax2.set_title("Y(0) Estimation")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Value")

        final_val = y0_data[-1]
        ax2.axhline(
            final_val,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Final: {final_val:.4f}",
        )
        if exact_y0 is not None:
            ax2.axhline(
                exact_y0,
                color="k",
                linestyle=":",
                alpha=0.7,
                label=f"Benchmark: {exact_y0:.4f}",
            )
        ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


def plot_trajectories_with_terminal(data, title, save_path):
    """Plot sample paths with terminal condition markers."""
    t = data["t"]
    Y = data["Y_learned"]
    Y_exact = data["Y_exact"]
    terminal_payoff = data.get("terminal_payoff")

    plt.figure(figsize=(8, 5))

    n_show = min(10, Y.shape[0])
    has_exact = Y_exact is not None and np.isfinite(Y_exact).any()
    for i in range(n_show):
        c = plt.cm.tab10(i % 10)
        plt.plot(t, Y[i], "-", color=c, alpha=0.6, linewidth=1.0)
        if has_exact and np.isfinite(Y_exact[i]).any():
            plt.plot(t, Y_exact[i], "--", color=c, alpha=0.4)

    plt.plot([], [], "k-", label="Learned")
    if has_exact:
        plt.plot([], [], "k--", label="Exact")
    if terminal_payoff is not None:
        plt.scatter(
            np.full(n_show, t[-1]),
            terminal_payoff[:n_show],
            marker="x",
            color="k",
            alpha=0.8,
            label="Payoff g(X_T)",
        )
        plt.scatter(
            np.full(n_show, t[-1]),
            _to_numpy(Y[:n_show, -1]),
            marker="o",
            color="tab:orange",
            alpha=0.7,
            label="Predicted Y_T",
        )

    plt.title(f"Solution Paths: {title}")
    plt.xlabel("Time t")
    plt.ylabel("Value u(t, X_t)")
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


def plot_terminal_fit(data, title, save_path):
    """Plot terminal condition fit and error histogram."""
    terminal_payoff = data.get("terminal_payoff")
    if terminal_payoff is None:
        return

    y_pred = _to_numpy(data["Y_learned"][:, -1])
    error = y_pred - terminal_payoff

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.scatter(terminal_payoff, y_pred, s=14, alpha=0.6)
    min_val = min(terminal_payoff.min(), y_pred.min())
    max_val = max(terminal_payoff.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.0)
    ax1.set_title("Terminal Fit")
    ax1.set_xlabel("Payoff g(X_T)")
    ax1.set_ylabel("Predicted Y_T")
    ax1.grid(True, alpha=0.3)

    ax2.hist(error, bins=20, color="tab:gray", alpha=0.8)
    ax2.set_title("Terminal Error")
    ax2.set_xlabel("Y_T - g(X_T)")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


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
    print(f"Saved greeks.png")
    plt.close()


def _load_benchmark_records(paths: List[str]) -> List[Dict]:
    records = []
    for path in paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Skipping benchmark file {path}: {exc}")
            continue

        if isinstance(data, dict) and "experiment" in data:
            exp = data["experiment"]
            if exp == "exp01_bsb_dimension":
                for row in data.get("results", []):
                    records.append(
                        {
                            "group": "BSB",
                            "label": f"D={row.get('dimension')}",
                            "pred": row.get("Y0_pred"),
                            "bench": row.get("Y0_exact"),
                            "bench_err": None,
                            "rel_error": row.get("relative_error_pct"),
                        }
                    )
            elif exp == "exp03_black_scholes":
                for row in data.get("results", []):
                    records.append(
                        {
                            "group": "BS",
                            "label": f"D={row.get('dimension')}",
                            "pred": row.get("Y0_pred"),
                            "bench": row.get("mc_price"),
                            "bench_err": row.get("mc_stderr"),
                            "rel_error": row.get("rel_error_pct"),
                        }
                    )
        elif isinstance(data, dict) and "true_value" in data and "pred_value" in data:
            records.append(
                {
                    "group": "HJB",
                    "label": f"D={data.get('dimension')}",
                    "pred": data.get("pred_value"),
                    "bench": data.get("true_value"),
                    "bench_err": None,
                    "rel_error": data.get("relative_error"),
                }
            )

    return records


def plot_benchmark_comparisons(records: List[Dict], save_dir: Path):
    if not records:
        return

    grouped: Dict[str, List[Dict]] = {}
    for r in records:
        grouped.setdefault(r["group"], []).append(r)

    for group, group_records in grouped.items():
        group_records = [r for r in group_records if r["pred"] is not None]
        if not group_records:
            continue

        labels = [r["label"] for r in group_records]
        pred = np.array([r["pred"] for r in group_records], dtype=float)
        bench = np.array([r["bench"] for r in group_records], dtype=float)
        rel_error = np.array([r["rel_error"] for r in group_records], dtype=float)
        bench_err = [r.get("bench_err") for r in group_records]

        x = np.arange(len(labels))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.bar(x - width / 2, pred, width, label="Deep BSDE")
        ax1.bar(x + width / 2, bench, width, label="Benchmark")
        if any(err is not None for err in bench_err):
            err_vals = [err if err is not None else 0.0 for err in bench_err]
            ax1.errorbar(
                x + width / 2,
                bench,
                yerr=err_vals,
                fmt="none",
                ecolor="k",
                elinewidth=1.0,
                capsize=3,
            )
        ax1.set_title(f"{group}: Price Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.bar(x, rel_error, color="tab:gray")
        ax2.set_title(f"{group}: Relative Error (%)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Relative Error (%)")
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = save_dir / f"comparison_{group.lower()}.png"
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()


def plot_single_price_comparison(
    pred: float,
    benchmark: float,
    benchmark_err: Optional[float],
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar([0, 1], [pred, benchmark], color=["tab:blue", "tab:orange"])
    if benchmark_err is not None:
        ax.errorbar([1], [benchmark], yerr=[benchmark_err], fmt="none", ecolor="k")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Deep BSDE", "Benchmark"])
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


def compute_price_surface_mc(
    eq: BlackScholesEquation,
    strike_grid: np.ndarray,
    spot_grid: np.ndarray,
    n_paths: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MC price surface over strikes and spot scales."""
    original_strike = eq.strike
    original_spot = eq.X0_val

    surface = np.zeros((len(spot_grid), len(strike_grid)))
    stderr = np.zeros_like(surface)

    for i, spot in enumerate(spot_grid):
        eq.X0_val = float(spot)
        for j, strike in enumerate(strike_grid):
            eq.strike = float(strike)
            price, err = eq.monte_carlo_price(n_paths=n_paths)
            surface[i, j] = price
            stderr[i, j] = err

    eq.strike = original_strike
    eq.X0_val = original_spot

    return surface, stderr


def plot_price_surface_heatmap(
    strike_grid: np.ndarray,
    spot_grid: np.ndarray,
    surface: np.ndarray,
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    mesh = ax.pcolormesh(
        strike_grid,
        spot_grid,
        surface,
        shading="auto",
        cmap="viridis",
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Spot S0")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="Price")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


def _strip_wrapper_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if any(k.startswith("net.") for k in state_dict.keys()):
        return {k.replace("net.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _infer_naisnet_architecture(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, int, int]:
    clean_state = _strip_wrapper_prefix(state_dict)
    first_weight = clean_state.get("first_layer.weight")
    output_weight = clean_state.get("output_layer.weight")
    if first_weight is None or output_weight is None:
        raise ValueError("Missing NAISNet weights for architecture inference.")

    hidden_dim, input_dim = first_weight.shape
    _, hidden_dim_out = output_weight.shape
    if hidden_dim_out != hidden_dim:
        hidden_dim = hidden_dim_out

    nais_weights = [
        k
        for k in clean_state.keys()
        if k.startswith("nais_layers.") and k.endswith(".weight")
    ]
    num_layers = len(nais_weights) + 1
    return input_dim, hidden_dim, num_layers


def _generate_xva_paths(solver, n_paths: int, n_steps: int) -> Dict[str, np.ndarray]:
    t, S, _ = solver.simulate_paths(n_paths, n_steps)
    Y_vals = []
    solver.net.eval()
    with torch.no_grad():
        for i, t_i in enumerate(t):
            t_batch = torch.full((n_paths, 1), t_i.item(), device=solver.device)
            S_batch = S[:, i : i + 1]
            Y_i = solver._forward(t_batch, S_batch).squeeze()
            Y_vals.append(Y_i)

    Y_learned = torch.stack(Y_vals, dim=1)
    terminal_payoff = solver.payoff(S[:, -1]).detach().squeeze()

    return {
        "t": t.cpu().numpy(),
        "Y_learned": Y_learned.cpu().numpy(),
        "Y_exact": None,
        "terminal_payoff": terminal_payoff.cpu().numpy(),
    }


def load_and_visualize_xva(args):
    device = get_device(args.device)
    model_path = Path(args.model)

    print(f"Visualizing Model: {model_path.name}")
    print("Type: XVA")

    print(f"Loading {args.model}...")
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"FATAL loading error: {exc}")
        return

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        print("FATAL loading error: Expected XVA checkpoint with 'model' key.")
        return

    state_dict = checkpoint["model"]
    try:
        _, hidden_dim, num_layers = _infer_naisnet_architecture(state_dict)
    except Exception as exc:
        print(f"FATAL loading error: {exc}")
        return

    try:
        from experiments.exp_xva import (
            XVABSDESolver,
            XVAParams as XVAParamsFull,
            plot_xva_results,
        )
    except Exception as exc:
        print(f"FATAL loading error: {exc}")
        return

    params = checkpoint.get("params") or XVAParamsFull()
    if not isinstance(params, XVAParamsFull):
        try:
            params = XVAParamsFull(**vars(params))
        except Exception:
            params = XVAParamsFull()

    solver = XVABSDESolver(
        params=params,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=str(device),
    )
    solver.net.load_state_dict(state_dict)
    solver.losses = checkpoint.get("losses", [])
    solver.prices = checkpoint.get("prices", [])
    solver.current_iteration = checkpoint.get("iteration", len(solver.losses))

    save_dir = Path(args.save_dir) / "xva"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_xva_results(solver, save_dir=str(save_dir))

    print("Done.")


# --- MAIN LOGIC ---
def load_and_visualize(args):
    device = get_device(args.device)
    model_path = Path(args.model)
    fname = model_path.name.lower()

    # 1. Infer Equation Type
    if args.equation is None:
        if "hjb" in fname:
            args.equation = "hjb"
        elif "bsb" in fname:
            args.equation = "bsb"
        elif "xva" in fname:
            args.equation = "xva"
        elif "bs" in fname:
            args.equation = "bs"
        else:
            args.equation = "bsb"  # Default

    # 2. Infer Dimension (Regex for D10, D100 etc)
    if args.dim == 100:  # Only override default
        match = re.search(r"d(\d+)", fname)
        if match:
            args.dim = int(match.group(1))
            print(f"Auto-detected dimension: {args.dim}")

    print(f"Visualizing Model: {model_path.name}")
    print(f"Type: {args.equation.upper()}, Dim: {args.dim}")

    if args.equation == "xva":
        load_and_visualize_xva(args)
        return

    # 3. Setup Equation & Network
    eq = get_equation(args.equation, args.dim, device)

    # 4. Robust Loading
    print(f"Loading {args.model}...")
    try:
        # We've already injected XVAParams at module level, so safe globals works
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            training_history = checkpoint.get("history", {})
        else:
            state_dict = checkpoint
            training_history = {}

        # Infer architecture from checkpoint to avoid shape mismatches.
        try:
            input_dim, hidden_dim, num_layers = _infer_naisnet_architecture(state_dict)
            inferred_dim = input_dim - 1
            if inferred_dim != args.dim:
                print(f"Overriding dim from {args.dim} to {inferred_dim} (checkpoint).")
                args.dim = inferred_dim
                eq = get_equation(args.equation, args.dim, device)
        except Exception:
            input_dim = args.dim + 1
            hidden_dim = 256
            num_layers = 4

        # Architecture inference
        if args.equation == "hjb":
            activation = "silu"
        else:
            activation = "sine"

        raw_net = NAISNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            activation=activation,
        )

        solver = StandardSolver(eq, raw_net, SolverConfig(), device=device)
        solver.training_history = training_history

        # Handle Wrappers
        is_wrapped = any(k.startswith("net.") for k in state_dict.keys())
        if is_wrapped:
            print("Detected Wrapped Model. Applying BlackScholesWrapper...")
            K = getattr(eq, "strike", float(args.dim))
            solver.model = BlackScholesWrapper(solver.model, eq.T, K).to(device)

        solver.model.load_state_dict(state_dict)
        print("Model loaded successfully.")

    except Exception as e:
        print(f"FATAL loading error: {e}")
        return

    # 5. Generate Plots
    save_dir = Path(args.save_dir) / args.equation
    save_dir.mkdir(parents=True, exist_ok=True)

    # Learning Curves
    exact_y0 = None
    if solver.training_history:
        if eq.has_exact_solution() and not isinstance(eq, HJBEquation):
            try:
                exact_y0 = (
                    eq.exact_solution(0.0, eq.sample_initial_condition(1))
                    .detach()
                    .item()
                )
            except Exception:
                exact_y0 = None
        plot_learning_history(
            solver.training_history,
            f"{args.equation.upper()} Training History (D={args.dim})",
            save_dir / "training_history.png",
            exact_y0=exact_y0,
        )

    # Trajectories
    print("Generating trajectories...")
    path_data = generate_paths_and_values(
        solver, n_paths=args.n_paths, n_steps=args.n_steps
    )
    plot_trajectories_with_terminal(
        path_data,
        f"{args.equation.upper()} Sample Paths (D={args.dim})",
        save_dir / "trajectories.png",
    )
    plot_terminal_fit(
        path_data,
        f"{args.equation.upper()} Terminal Fit (D={args.dim})",
        save_dir / "terminal_fit.png",
    )

    # Greeks (BS / BSB only)
    if args.equation in ["bs", "bsb"]:
        plot_bs_greeks(
            solver,
            f"{args.equation.upper()} Greeks (D={args.dim})",
            save_dir,
            delta_mode=args.delta_mode,
            mc_paths=args.greeks_mc_paths,
            spot_min=args.spot_min,
            spot_max=args.spot_max,
            spot_steps=args.spot_steps,
        )

    # Single-model price comparison (exact or MC)
    if args.price_compare:
        pred_price = solver.predict(0.0, eq.sample_initial_condition(1)).item()
        bench_price = None
        bench_err = None
        if eq.has_exact_solution() and not isinstance(eq, HJBEquation):
            try:
                bench_price = (
                    eq.exact_solution(0.0, eq.sample_initial_condition(1))
                    .detach()
                    .item()
                )
            except Exception:
                bench_price = None
        elif isinstance(eq, HJBEquation) and args.exact_mc > 0:
            bench_price = (
                eq.exact_solution(
                    0.0, eq.sample_initial_condition(1), n_mc=args.exact_mc
                )
                .detach()
                .item()
            )
        elif hasattr(eq, "monte_carlo_price"):
            bench_price, bench_err = eq.monte_carlo_price(n_paths=args.mc_paths)

        if bench_price is not None:
            plot_single_price_comparison(
                pred_price,
                bench_price,
                bench_err,
                f"{args.equation.upper()} Price Comparison (D={args.dim})",
                save_dir / "price_comparison.png",
            )

    # Benchmark summary plots from JSON files
    if args.benchmark_json:
        records = _load_benchmark_records(args.benchmark_json)
        plot_benchmark_comparisons(records, save_dir)

    # Price surface / heatmap for BS basket options
    if args.surface and isinstance(eq, BlackScholesEquation):
        strike_grid = np.linspace(
            args.strike_min * eq.strike, args.strike_max * eq.strike, args.strike_steps
        )
        spot_grid = np.linspace(
            args.spot_min * eq.X0_val, args.spot_max * eq.X0_val, args.spot_steps
        )
        surface, _ = compute_price_surface_mc(
            eq, strike_grid, spot_grid, n_paths=args.surface_mc_paths
        )
        plot_price_surface_heatmap(
            strike_grid,
            spot_grid,
            surface,
            f"{args.equation.upper()} MC Price Surface (D={args.dim})",
            save_dir / "price_surface.png",
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--equation", type=str, default=None, choices=["bsb", "bs", "hjb", "xva"]
    )
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="results/figures")
    parser.add_argument("--n-paths", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument(
        "--delta-mode",
        type=str,
        default="mean",
        choices=["sum", "mean"],
        help="Delta aggregation across assets",
    )
    parser.add_argument(
        "--greeks-mc-paths",
        type=int,
        default=0,
        help="MC paths for BS benchmark curve (0 = skip)",
    )
    parser.add_argument(
        "--benchmark-json",
        type=str,
        nargs="*",
        default=[],
        help="Benchmark JSON files (exp01/exp03/hjb) for comparison plots",
    )
    parser.add_argument(
        "--price-compare",
        action="store_true",
        help="Generate a single price comparison plot (model vs benchmark)",
    )
    parser.add_argument(
        "--mc-paths",
        type=int,
        default=200000,
        help="MC paths for benchmarks when needed",
    )
    parser.add_argument(
        "--exact-mc",
        type=int,
        default=0,
        help="MC paths for HJB exact solution (0 = skip)",
    )
    parser.add_argument(
        "--surface",
        action="store_true",
        help="Generate MC price surface/heatmap for BS basket options",
    )
    parser.add_argument("--spot-min", type=float, default=0.5)
    parser.add_argument("--spot-max", type=float, default=1.5)
    parser.add_argument("--spot-steps", type=int, default=25)
    parser.add_argument("--strike-min", type=float, default=0.5)
    parser.add_argument("--strike-max", type=float, default=1.5)
    parser.add_argument("--strike-steps", type=int, default=25)
    parser.add_argument("--surface-mc-paths", type=int, default=50000)

    args = parser.parse_args()
    load_and_visualize(args)
