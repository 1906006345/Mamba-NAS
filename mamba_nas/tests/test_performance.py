import torch
from torch import nn

from mamba_nas.performance import model_performance


def test_model_performance_reports_trainable_parameters_and_forward_macs():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, values, *_):
            return self.linear(values)

    model = Toy()
    loader = [(torch.zeros(1, 4, 3), torch.zeros(1, dtype=torch.long), torch.ones(1, 4))]
    metrics = model_performance(model, loader, torch.device("cpu"))
    assert metrics["parameters"] == 8
    assert metrics["macs"] > 0
    assert metrics["flops"] == metrics["macs"] * 2
    assert "profile_flops" in metrics
    assert "latency_median_ms" in metrics
    assert "FLOPs / 2" in metrics["macs_definition"]
