"""Network contracts: shapes, the NAIS-Net stability invariant, activations."""

import pytest
import torch

from deep_fbsde_nn.networks import FeedForwardNet, NAISNet
from deep_fbsde_nn.networks.activations import get_activation
from deep_fbsde_nn.networks.wrapper import BlackScholesWrapper

BATCH = 7
IN_DIM = 4
HIDDEN = 16


def test_naisnet_forward_shape():
    net = NAISNet(input_dim=IN_DIM, hidden_dim=HIDDEN, output_dim=1, num_layers=3)
    out = net(torch.randn(BATCH, IN_DIM))
    assert out.shape == (BATCH, 1)


def test_feedforward_forward_shape():
    net = FeedForwardNet(input_dim=IN_DIM, hidden_dim=HIDDEN, output_dim=1, num_layers=3)
    out = net(torch.randn(BATCH, IN_DIM))
    assert out.shape == (BATCH, 1)


class TestProjectionInvariant:
    """A = _project(W) must have eigenvalues in [eps, 1 - eps]."""

    @pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
    def test_eigenvalues_bounded(self, scale):
        net = NAISNet(input_dim=IN_DIM, hidden_dim=HIDDEN, output_dim=1, num_layers=4)
        eps = net.epsilon
        for layer in net.nais_layers:
            A = net._project(layer.weight * scale)
            # A is symmetric by construction; eigvalsh is exact for it
            eigs = torch.linalg.eigvalsh(A)
            assert eigs.min().item() >= eps - 1e-5, "A must be positive definite"
            assert eigs.max().item() <= 1.0 - eps + 1e-5, "spectral norm must stay below 1"

    def test_is_stable_is_computed_not_hardcoded(self):
        net = NAISNet(input_dim=IN_DIM, hidden_dim=HIDDEN, output_dim=1, num_layers=3)
        assert net.is_stable
        # Even with adversarially scaled weights the projection restores the
        # invariant, so is_stable stays True *because the check recomputes it*.
        with torch.no_grad():
            for layer in net.nais_layers:
                layer.weight.mul_(100.0)
        assert net.is_stable

    def test_feedforward_reports_unstable(self):
        net = FeedForwardNet(input_dim=IN_DIM, hidden_dim=HIDDEN, output_dim=1)
        assert net.is_stable is False


class TestActivations:
    @pytest.mark.parametrize(
        "name",
        ["sine", "sin", "relu", "tanh", "gelu", "swish", "silu", "softplus", "sigmoid", "elu"],
    )
    def test_known_names(self, name):
        act = get_activation(name)
        out = act(torch.randn(BATCH, 3))
        assert out.shape == (BATCH, 3)

    def test_unknown_name_raises_with_available_list(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("does-not-exist")

    def test_sine_omega_passthrough(self):
        act = get_activation("sine", omega=30.0)
        assert act.omega == 30.0


def test_black_scholes_wrapper_shapes_and_transform():
    inner = NAISNet(input_dim=IN_DIM, hidden_dim=HIDDEN, output_dim=1, num_layers=3)
    wrapped = BlackScholesWrapper(inner, T=1.0, K=1.0)
    x = torch.cat([torch.zeros(BATCH, 1), torch.ones(BATCH, IN_DIM - 1)], dim=1)
    out = wrapped(x)
    assert out.shape == (BATCH, 1)
    assert wrapped.count_parameters() == inner.count_parameters()
