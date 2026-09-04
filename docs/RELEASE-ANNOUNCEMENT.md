# Release Announcement Plan

**Fires at v0.2 (the benchmark release), not v0.1.** v0.1 tags quietly: it
starts the maintained-clock and mints the DOI; the announcement spends the
one-shot channels on the strongest story — the reproducible benchmark table
plus working XVA. Decided in CEO review, 2026-09-04.

## Before announcing (checklist)

- [ ] Demand-evidence check (T21, ~1 hour): issue/fork traffic on the four
      unmaintained PyTorch deep-BSDE repos, citation velocity of Han et al.
      2018 / Güler et al. 2019, `deepxde` download counts as an audience
      ceiling. Summarize in one paragraph below.
- [ ] v0.2 benchmark table regenerated and in the README.
- [ ] Hero figure regenerated post-benchmark (`python experiments/make_hero_figure.py`).
- [ ] Activate GitHub Discussions + post the seeded good-first-issues.
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

## Draft release note (rewrite with benchmark numbers before firing)

> **deep-fbsde-nn — a tested, citable PyTorch library for Deep BSDE methods**
>
> The canonical Deep BSDE implementations are TensorFlow research code from
> 2018. This is a maintained PyTorch library: `pip install deep-fbsde-nn`,
> core deps torch+numpy only, every equation with a known reference validated
> in CI, benchmark table regenerable with one command, DOI for citation.
>
> Built from a CQF final project on the stability-conditioned NAIS-Net
> variant of Güler, Laignelet & Parpas (2019). Honest caveats in the README
> (which numerical claims CI proves, and which it doesn't yet).

## Demand-evidence summary

_(fill in from T21 before announcing)_
