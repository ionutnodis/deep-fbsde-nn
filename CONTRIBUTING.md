# Contributing

Thanks for considering a contribution. This is a solo-maintained research library; issues and PRs are read, reviews are best-effort (see the support policy in the README).

## Dev setup

```bash
git clone https://github.com/ionutnodis/deep-fbsde-nn.git
cd deep-fbsde-nn
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # core + pytest, ruff, build tooling
pip install -e ".[experiments]"   # only if you run experiments/ scripts
```

The experiment scripts bootstrap `sys.path` so they also run from a bare clone, but an editable install is the supported path.

## Checks that must pass

```bash
ruff check .            # lint (config in pyproject.toml)
pytest -m "not slow"    # fast suite, runs per-push in CI (~seconds)
pytest -m slow          # convergence suite, runs on tag + weekly (~minutes)
```

Every code path needs a test. Correctness claims need a reference: an exact solution, a closed form, or a seeded Monte-Carlo benchmark (see `tests/test_convergence.py` for the pattern, including the PDE-residual technique for validating a driver against a reference solution).

## Conventions worth knowing

- **Driver sign:** `driver()` returns the `f` of the PDE form `∂t u + μ·∇u + ½Tr(σσᵀD²u) + f = 0`; both solvers integrate `dY = -f dt + Zᵀσ dW`. `Z` is the raw gradient `∇u`.
- **Lean core:** the installed package imports only `torch` and `numpy`. Plotting and scipy belong in `experiments/`.
- **Checkpoints** must stay loadable under `torch.load(weights_only=True)` — tensors and Python primitives only.
- **Experimental code** lives in `experiments/experimental/` and warns at import.

## Good first issues

- `net_u` passes `retain_graph=True` unconditionally (`solvers/base.py`) — only needed while training.
- NAISNet/FeedForwardNet duplicate `count_parameters`/`__repr__` — extract a shared base.
- Add a new equation subclass with a reference solution (a guided template for exactly this is in the README's "Extending the Library").

## Releases (maintainer)

Tag `v*` → CI runs the slow suite → `publish.yml` builds, dry-runs on TestPyPI, publishes to PyPI via trusted publishing → Zenodo archives the release. Regenerate the hero figure (`python experiments/make_hero_figure.py`) before tagging.
