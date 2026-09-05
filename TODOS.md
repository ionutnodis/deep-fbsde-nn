# TODOS

Deferred work with context. Generated during /plan-eng-review of the v0.1 release plan (2026-09-04).

## 0.2 candidates

### Promote XVA into the public API (0.3 headline)
- **What:** `XVAEquation(BaseEquation)` in the package with the Burgard-Kjaer driver, priced via library solvers; `GlobalSolver` backing for the full price/greeks surface (fixes the far-from-spot delta gap the 0.2 validation documented); retire the bespoke loop in `experiments/exp_xva.py` to a thin driver script.
- **Why:** "Price and risk-manage XVA with the library" becomes real API instead of an experiment; GlobalSolver was built for exactly this surface-learning job.
- **Context:** 0.2 validated the experiment in place (price 4.1%, components tight vs classical MC — tests/test_xva.py). Chosen as option C in the 0.2 decision round: validate now, promote in 0.3 with its own design pass.
- **Depends on:** 0.2 shipped.

### Benchmark-first 0.2 release — now also the launch event
- **What:** One command regenerates an equation × dimension (10/50/100/200) table of relative error + runtime, published in the README; loosened-tolerance version doubles as a regression suite. Per the CEO-review outside voice (2026-09-04): v0.1 tags quietly, and 0.2 is when the announcement fires, the JOSS submission goes in, and Discussions + seeded good-first-issues activate — the benchmark is the story, and the one-shot channels (Show HN) are spent on it.
- **Why:** Reproducible numbers are the strongest credibility signal in this niche and give researchers something to compare against.
- **Pros:** Marketing and regression suite in one artifact.
- **Cons:** Benchmark runs are slow; needs a cut-down CI variant and result-presentation work.
- **Context:** Office-hours approach C, deferred not rejected. `experiments/exp01_bsb_dimension.py` and `tables.py` already do most of the sweep.
- **Depends on:** v0.1 convergence suite (tolerances calibrated).

### Docs site
- **What:** mkdocs-material (or similar) site: API reference from docstrings, math background, equation-authoring guide.
- **Why:** README + docstrings are the 0.1 documentation surface (also the JOSS fallback position). A site becomes worth it when the API surface or contributor count grows.
- **Context:** Deferred in CEO review; JOSS reviewers may pull minor docs work into 0.1 — accept reviewer-driven additions without building the site.
- **Depends on:** nothing hard; best after 0.2 benchmark tables exist to show off.

### Minor cleanups (seeded as a good-first-issue at launch)
- **What:** NAISNet/FeedForwardNet duplicate `count_parameters`/`__repr__`/activation resolution (candidate shared base class). (`retain_graph` fixed in 0.2.)
- **Why:** DRY; kept deliberately as a community on-ramp.
- **Cons:** Pure refactor, no user-visible change — lowest priority.
- **Depends on:** test suite green (Stage 2) so the refactor is safe.
