## What & why

<!-- one paragraph; link the issue if there is one -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `pytest -m "not slow"` passes
- [ ] New code paths have tests; numerical claims have a reference
      (exact solution, closed form, or seeded MC benchmark)
- [ ] Core package still imports only torch + numpy
- [ ] Checkpoints still load under `torch.load(weights_only=True)`
