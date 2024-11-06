import torch 
import torch.nn as nn

from .coupling import AdditiveCouplingLayer
from .prior import LogisticDistribution
from .scaling import ScalingLayer

class NICE (nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 5, n_couplings: int = 4):
        super().__init__()
        self.n_couplings = n_couplings

        couplings   = [
            AdditiveCouplingLayer(i, input_size, hidden_size, n_layers)
            for i in range(self.n_couplings)
        ]

        self.couplings = nn.ModuleList(couplings)
        self.scaling = ScalingLayer(input_size)
        self.prior   = LogisticDistribution(input_size)

    def forward(self, x):
        for i in range(len(self.couplings)):
            x = self.couplings[i](x)
        x, det = self.scaling(x, 0)
        return x, torch.sum(self.prior.log_prob(x), axis = 1) + det

    def inverse(self, y):
        h, _ = self.scaling.inverse(y, 0)
        for layer in reversed(list(self.couplings)):
            h = layer.inverse(h)
        return h

    def sample(self, size: int):
        samples = self.prior.sample(size, None)
        return self.inverse(samples)