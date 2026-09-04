# Security Policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability) rather than a public issue. Reports get
priority over all other work; expect an acknowledgement within a few days.

## Scope notes

- Model checkpoints load with `torch.load(weights_only=True)`: loading an
  untrusted checkpoint must not execute code. If you find a way around
  that, it is exactly the kind of report we want.
- The `experiments/experimental/` directory is unvalidated research code and
  outside the supported surface.
