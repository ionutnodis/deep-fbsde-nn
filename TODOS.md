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
