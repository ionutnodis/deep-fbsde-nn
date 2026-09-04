"""Shared fixtures: deterministic seeds and CPU-only devices."""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    """Every test starts from the same RNG state (CPU streams)."""
    torch.manual_seed(0)
    np.random.seed(0)
    yield


@pytest.fixture()
def device():
    return torch.device("cpu")
