---
title: 'deep-fbsde-nn: A PyTorch library for solving high-dimensional PDEs with the Deep BSDE method'
tags:
  - Python
  - PyTorch
  - partial differential equations
  - backward stochastic differential equations
  - deep learning
  - quantitative finance
authors:
  - name: Ionut Nodis
    orcid: 0000-0000-0000-0000  # TODO: fill in before submission
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 2026-09-04  # updated at submission (post v0.2 — see docs/RELEASE-ANNOUNCEMENT.md)
bibliography: paper.bib
---

<!-- JOSS draft, submission-ready with v0.2. Remaining TODO: the author's
     ORCID in the frontmatter. Repository: github.com/ionutnodis/deep-fbsde-nn. -->

# Summary

Semilinear parabolic partial differential equations (PDEs) in tens to
hundreds of dimensions arise throughout stochastic control and mathematical
finance, where grid-based numerical methods are unusable. The Deep BSDE
method [@Han2018] reformulates such PDEs as forward-backward stochastic
differential equations (FBSDEs) and trains a neural network to satisfy the
discretized backward dynamics. `deep-fbsde-nn` packages this method as a
maintained, tested PyTorch library: a four-method equation interface
(drift, diffusion, driver, terminal), two solvers (fixed and distributed
initial conditions), the stability-conditioned NAIS-Net architecture
[@Ciccone2018; @Guler2019] whose block state matrices are spectrally
projected to eigenvalues in $[\varepsilon, 1-\varepsilon]$, and Multi-Level
Monte Carlo progressive time-stepping.

# Statement of need

The canonical Deep BSDE implementations — the reference TensorFlow code of
@Han2018 and the FBSNN codebase of @Raissi2018 — are research artifacts:
unmaintained, unpackaged, and untested. Several PyTorch reproductions exist
on GitHub in the same state. No installable, continuously tested library
serves researchers who want to use, extend, or benchmark against the
method rather than re-implement it. `deep-fbsde-nn` fills that gap with a
lean core (torch and numpy only), CI that validates every equation with a
known reference (exact solutions for Black-Scholes-Barenblatt and the
single-asset call; the published branching-diffusion value 0.052802 for the
canonical d=100 Allen-Cahn case; a seedable Cole-Hopf Monte-Carlo reference
for Hamilton-Jacobi-Bellman, checked both end-to-end and via a
finite-difference PDE-residual test of the implemented driver; a seeded
Monte-Carlo benchmark for basket options), a one-command reproducible
benchmark (11 seeded equation-by-dimension rows, nine below 1% relative
error on CPU, including the published d=100 HJB value 4.5901 reproduced to
0.03%), an XVA pricing example validated against a classical Monte-Carlo
oracle, and archived, citable releases.

# Acknowledgements

The implementation builds upon example code and guidance provided by
Panos Parpas (Imperial College London) in the context of the author's CQF
final project.

# References
