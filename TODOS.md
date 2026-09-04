# TODOS

Deferred work with context. Generated during /plan-eng-review of the v0.1 release plan (2026-09-04).

## 0.2 candidates

### Debug and promote XVA + greeks
- **What:** Fix the known-broken XVA experiment (`experiments/experimental/exp_xva.py`, 1,172 lines) and greeks plotting; promote them into the tested surface.
- **Why:** They are the most finance-distinctive content in the repo and the natural 0.2 headline ("price and risk-manage with deep BSDEs").
- **Pros:** Differentiates the library beyond a reference implementation; closest to the CQF story.
- **Cons:** Debugging effort is the least estimable in the repo (commit e39091a: "Debugging required for XVA's and greeks").
- **Context:** Quarantined as experimental in v0.1 (premise 2 of the release plan). Start from the module-level warning headers added in Stage 1.
- **Depends on:** v0.1 shipped; test infrastructure from Stage 2.

### Benchmark-first 0.2 release — now also the launch event
- **What:** One command regenerates an equation × dimension (10/50/100/200) table of relative error + runtime, published in the README; loosened-tolerance version doubles as a regression suite. Per the CEO-review outside voice (2026-09-04): v0.1 tags quietly, and 0.2 is when the announcement fires, the JOSS submission goes in, and Discussions + seeded good-first-issues activate — the benchmark is the story, and the one-shot channels (Show HN) are spent on it.
- **Why:** Reproducible numbers are the strongest credibility signal in this niche and give researchers something to compare against.
- **Pros:** Marketing and regression suite in one artifact.
- **Cons:** Benchmark runs are slow; needs a cut-down CI variant and result-presentation work.
- **Context:** Office-hours approach C, deferred not rejected. `experiments/exp01_bsb_dimension.py` and `tables.py` already do most of the sweep.
- **Depends on:** v0.1 convergence suite (tolerances calibrated).

### Evaluate the `batch_size=1` default
- **What:** Benchmark M=1 (current default, follows the Parpas reference — `deep_fbsde_nn/solvers/base.py:39`) against Han et al.'s 64-256 across the standard equations; change the default if the evidence says so.
- **Why:** A self-styled reference implementation whose defaults diverge from the canonical papers invites scrutiny; but changing it mid-release would have altered training behavior and invalidated tolerance calibration.
- **Pros:** Either outcome is a defensible, documented default backed by data.
- **Cons:** A default change is a behavior change for existing users; needs a minor-version bump and release-note callout.
- **Context:** Outside-voice finding #11 in the v0.1 eng review; v0.1 documents the M=1 choice in the README instead.
- **Depends on:** benchmark harness (item above) makes this nearly free.

### `setup_device` opt-in API
- **What:** Stop `utils/device.py:setup_device` mutating global torch state (`torch.set_num_threads(8)`, cudnn benchmark, TF32 flags) as a side effect; make optimizations opt-in parameters.
- **Why:** Library utility functions changing process-global state silently is a footgun for users embedding the library.
- **Pros:** Predictable library behavior; keeps benchmarks honest.
- **Cons:** Small API change; needs deprecation note.
- **Context:** Flagged in v0.1 eng review (code quality); v0.1 adds a docstring warning only.
- **Depends on:** nothing; bundle with any 0.2 API pass.

### Example notebook with execution harness + Colab badge
- **What:** `examples/quickstart_bsb.ipynb` executed in CI (nbclient/papermill) so it cannot rot, plus an "Open in Colab" README badge. Generate the notebook from the tested `examples/quickstart.py` via jupytext so the source of truth stays tested.
- **Why:** Notebooks are the lingua franca for research onboarding; Colab removes the install barrier entirely. But an unexecuted notebook breaks within two releases, so the execution harness is a precondition of the badge.
- **Context:** v0.1 ships tested `examples/quickstart.py` instead (eng review outside-voice finding #8; Colab badge deferred in CEO review D2.3 — "adds a moving part the day before release").
- **Depends on:** CI from Stage 4.

### Reference solutions for AllenCahn and the BS basket
- **What:** `AllenCahnEquation` and `BlackScholesEquation` (basket) ship in 0.1 with shape tests only — no reference to validate against. Add an MC benchmark for the basket and literature reference values for AllenCahn (e.g. the d=100 value from Han-Jentzen-E 2018), wiring both into the convergence suite.
- **Why:** Closes the last validation gap: after this, every exported equation is validated, not just the three with cheap references (BSB, D=1 vanilla call, HJB).
- **Context:** Surfaced by CEO-review spec loop (iteration 1, issue 1). Note: AllenCahn isn't even listed in the README's equation table today — the Stage 1 README truth pass should add it with an honest "no reference solution yet" marker.
- **Depends on:** convergence-suite infrastructure from Stage 2.

### Docs site
- **What:** mkdocs-material (or similar) site: API reference from docstrings, math background, equation-authoring guide.
- **Why:** README + docstrings are the 0.1 documentation surface (also the JOSS fallback position). A site becomes worth it when the API surface or contributor count grows.
- **Context:** Deferred in CEO review; JOSS reviewers may pull minor docs work into 0.1 — accept reviewer-driven additions without building the site.
- **Depends on:** nothing hard; best after 0.2 benchmark tables exist to show off.

### Polish pack
- **What:** `--cite` helper printing the BibTeX, informative solver `__repr__`, OOM error hint suggesting smaller batch, ETA in training printout.
- **Why:** Small touches that make researchers smile; each is <30 min.
- **Context:** Presented in CEO review, no decision taken — re-propose at 0.2.
- **Depends on:** nothing.

### Tighten HJB end-to-end convergence via per-timestep Z networks
- **What:** The single-network architecture derives Z = ∇u by autograd from the same network as u. On strongly nonlinear (quadratic-in-Z) drivers like HJB, the optimizer damps the martingale term by under-estimating ‖Z‖, which under-weights the driver: calibrated runs plateau ~12-15% above the Cole-Hopf reference at D=3 regardless of budget, network size, or time-grid resolution (verified during v0.1 test calibration). Implement the Han et al. original design (separate Z subnetwork per timestep) or an FBSNN-style variant as an alternative solver, then tighten `test_hjb_end_to_end_sanity` from 20% to ~5%.
- **Why:** The sharp correctness test (PDE-residual, `test_hjb_driver_consistent_with_reference`) already guards the math; this closes the end-to-end gap and makes HJB a headline validation instead of a caveat.
- **Context:** Discovered while calibrating v0.1 convergence tests — the same calibration that caught the StandardSolver driver-sign bug and the HJB driver factor-2 bug. See tests/test_convergence.py module docstring.
- **Depends on:** v0.1 test infrastructure (shipped).

### Minor cleanups
- **What:** `net_u` passes `retain_graph=True` unconditionally (`solvers/base.py:179`) — only needed during training; and NAISNet/FeedForwardNet duplicate `count_parameters`/`__repr__`/activation resolution (candidate shared base class).
- **Why:** Memory hygiene during long validation loops; DRY.
- **Cons:** Pure refactor, no user-visible change — lowest priority.
- **Depends on:** test suite green (Stage 2) so the refactor is safe.
