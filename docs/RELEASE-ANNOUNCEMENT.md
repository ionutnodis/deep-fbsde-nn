# Release Announcement Plan

**Fires at v0.2 (the benchmark release), not v0.1.** v0.1 tags quietly: it
starts the maintained-clock and mints the DOI; the announcement spends the
one-shot channels on the strongest story — the reproducible benchmark table
plus working XVA. Decided in CEO review, 2026-09-04.

## Before announcing (checklist)

- [x] Demand-evidence check (T21) — see summary below.
- [x] v0.2 benchmark table in the README (11 rows, 9 under 1%).
- [x] Hero figure current (seeded d=100 run, 2.31%).
- [ ] Activate GitHub Discussions + post the seeded good-first-issues (fires with the announcement).
- [ ] Submit the JOSS paper (benchmark = contribution evidence).

## Channels (research-first ordering)

1. SciML / ML-for-PDE communities (r/MachineLearning, the SciML discourse,
   ML-for-quant-finance mailing lists)
2. Show HN — one shot; title around the benchmark, not the release
3. r/quant + Wilmott forums (XVA angle)
4. CQF alumni network
5. LinkedIn (personal post)

## Post-launch health check

Watch for two weeks: PyPI downloads (pypistats.org/packages/deep-fbsde-nn),
GitHub traffic (Insights → Traffic), stars/issues. These numbers size the
0.3 investment.

## Release note (final — v0.2 numbers)

> **deep-fbsde-nn — a tested, citable PyTorch library for Deep BSDE methods**
>
> The canonical Deep BSDE implementations are TensorFlow research code from
> 2018 (296 stars, 135 forks, nothing installable). This is a maintained
> PyTorch library: `pip install deep-fbsde-nn`, core deps torch+numpy only.
>
> The receipts, reproducible with one command (`python benchmarks/run.py`):
> an 11-row seeded benchmark across every shipped equation — 9 rows under 1%,
> including Han et al.'s published d=100 HJB value 4.5901 reproduced to
> 0.03% and the d=100 Allen-Cahn branching-diffusion value to 0.98%, on a
> laptop CPU. XVA pricing validated against a classical Monte-Carlo oracle.
> One-click Colab quickstart, executed in CI so it can't rot. DOI for citation.
>
> Built from a CQF final project on the stability-conditioned NAIS-Net
> variant of Güler, Laignelet & Parpas (2019). Honest caveats in the README
> (exactly which numerical claims CI proves, and which it doesn't yet).

## Demand-evidence summary

Measured 2026-09-05 via GitHub API: the canonical TF repos hold real interest
with zero product — frankhan91/DeepBSDE 296 stars / 135 forks (people fork
research code because nothing is installable), FBSNNs 162 stars (dead since
2020). The four PyTorch reproductions total ~86 stars, all stale. deepxde
(adjacent SciML ceiling): 4,408 stars. Verdict: a real niche of hundreds of
active researchers; calibrate success to tens-of-stars near-term, not front
pages — research-first channels accordingly.
