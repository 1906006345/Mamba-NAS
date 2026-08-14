from __future__ import annotations

import torch
from torch import nn


class FakeMixer(nn.Module):
    """CPU-test mixer with temporal propagation and the same input/output contract."""

    def __init__(self, dimension: int):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension, bias=False)

    def forward(self, x):
        cumulative = torch.cumsum(x, dim=1)
        denominator = torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        return self.projection(cumulative / denominator)


def fake_mixer_factory(genome):
    return FakeMixer(genome.d_model)

