# TODOS

Deferred work with context. Generated during /plan-eng-review of the v0.1 release plan (2026-09-04).

## 0.2 candidates

### Single-network pricing surface (research)
- **What:** Learn u(t,S) across wide moneyness in one network. Blocked by a multi-scale loss floor: value²-weighted squared error starves OTM wings, relative reweighting destroys the body (both measured during 0.3), and the terminal-gradient target is discontinuous at the strike.
- **Context:** 0.3 shipped the practical alternative instead — anchored pointwise `StepwiseSolver` evaluation (delta ≤0.011 across S=75-130). Candidate directions: control variates with better structure, strike-smoothed terminal gradients, per-region networks.
- **Depends on:** nothing; research item.

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
