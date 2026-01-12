import torch
import torch.nn as nn


class BlackScholesWrapper(nn.Module):
    """
    Wraps a raw neural network to handle financial input scaling.

    Transforms inputs:
        t -> Time to maturity (T - t)
        S -> Log-moneyness log(S / K)

    This ensures the network sees inputs centered around 0 with scale ~1,
    preventing saturation and 'dead' deltas.
    """

    def __init__(self, net, T: float, K: float):
        super().__init__()
        self.net = net
        self.T = T
        self.K = K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, D + 1).
               x[:, 0] is time t
               x[:, 1:] is asset prices S
        """
        t = x[:, 0:1]
        S = x[:, 1:]

        # 1. Time to Maturity (normalized by T is optional, but T-t is crucial)
        tau = self.T - t

        # 2. Log-Moneyness
        # Add epsilon to prevent log(0)
        log_moneyness = torch.log((S + 1e-6) / self.K)

        # Concatenate: [tau, log_S1, log_S2, ...]
        features = torch.cat([tau, log_moneyness], dim=1)

        return self.net(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
