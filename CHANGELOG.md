# Changelog

## 0.2.0 — 2026-09-05

The benchmark release: every equation validated, every number reproducible.

### Added
- **`StepwiseSolver`** — the original Han-Jentzen-E (2018) parameterization
  (trainable Y₀/Z₀ + one Z-network per timestep). Reproduces the published
  d=100 HJB value 4.5901 to **0.03%** on CPU. Two automatic stability
  mechanisms, no per-problem tuning: Y₀ warm-started at the driver-free
  value E[g(X_T)] and zero-initialized Z-net outputs.
- **Benchmark harness** (`python benchmarks/run.py`): 11-row seeded
  equation × dimension table (rel. error + wall time), spliced into the
  README. 9 of 11 rows under 1%.
- **Reference validation for every exported equation**, including the
  published branching-diffusion value for Allen-Cahn (0.052802 at d=100,
  reproduced to 0.98%) and a seeded MC benchmark for the BS basket (0.62%).
- **XVA validated against the classical Monte-Carlo oracle** (price 4.1%;
  CVA 2.4%, FVA 0.7%, DVA correctly ~0). `experiments/exp_xva.py` graduates
  out of quarantine.
- **Colab quickstart notebook**, generated from a jupytext source that CI
  executes on every push (badge in the README).
- `python -m deep_fbsde_nn --cite`, informative solver `__repr__`s, OOM
  errors re-raised with batch/timestep guidance, ETA in training printouts.
- Slow test for the HJB driver via a finite-difference PDE-residual check of
  the Cole-Hopf reference (sharp discriminator: residual 4e-4 correct vs
  1.7e-2 / 6.6e-2 for the two historical driver bugs).

### Changed (behavior)
- **`AllenCahnEquation` re-specified to the canonical literature benchmark**
  (σ=√2, g(x)=1/(2+0.4‖x‖²), analytic terminal gradient). The 0.1 variant
  had no reference value anywhere and was unvalidatable.
- **`SolverConfig.batch_size` default 1 → 16** after an evidence sweep
  (BSB d=2: 3.61% → 1.89%; vanilla D=1: 1.09% → 0.05% at ~1.2x cost).
  Note: `StandardSolver`'s loss sums over the batch, so batch size scales
  the effective learning rate.
- **`setup_device` is now a read-only probe**; global torch optimizations
  (cudnn benchmark, TF32, thread count) are opt-in via `optimize=True`.
- `net_u` retains autograd graphs only during training.

### Fixed
- **XVA driver documentation** matched the historical wrong sign convention
  in four docstrings (the code was already correct); the missing per-step
  consistency loss (previously a stub) left the network's t>0 level
  unconstrained, corrupting the XVA component breakdown — implemented,
  mirroring `StandardSolver`.
- Off-spot XVA inputs now trained on a mixed batch (half at S₀, half on a
  ±20% band). At-the-money delta is accurate; far-from-spot greeks remain
  documented as the 0.3 target.

## 0.1.0 — 2026-09-04

First public release: installable, tested, citable.

- Lean core (torch + numpy), safe checkpoints (`weights_only=True`),
  spectrally conditioned NAIS-Net with a computed stability invariant.
- 77 tests; CI with CPU torch wheels; PyPI trusted publishing; Zenodo DOI.
- Two solver-math fixes found by the new test suite before anyone hit them:
  a driver-sign convention mismatch between the two solvers, and a factor-2
  error in the HJB driver.
