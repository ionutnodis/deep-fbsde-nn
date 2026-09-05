# TODOS

Deferred work with context. Generated during /plan-eng-review of the v0.1 release plan (2026-09-04).

## Open

### Single-network pricing surface (research)
- **What:** Learn u(t,S) across wide moneyness in one network. Blocked by a multi-scale loss floor: value²-weighted squared error starves OTM wings, relative reweighting destroys the body (both measured during 0.3), and the terminal-gradient target is discontinuous at the strike.
- **Context:** 0.3 shipped the practical alternative instead — anchored pointwise `StepwiseSolver` evaluation (delta ≤0.011 across S=75-130). Candidate directions: control variates with better structure, strike-smoothed terminal gradients, per-region networks.
- **Depends on:** nothing; research item.

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

## Completed

### Benchmark-first 0.2 release — now also the launch event
- **What:** One command regenerates an equation × dimension table of relative error + runtime, published in the README; loosened-tolerance version doubles as a regression suite (`python benchmarks/run.py`).
- **Completed:** v0.2.0 (2026-09-05). The 0.3.0 release added the XVA row (12 rows, 10 under 1%).
