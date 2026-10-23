from types import SimpleNamespace

import torch
from torch import nn

import models.MambaBaseline as mamba_baseline


class FakeMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)
        self.settings = (d_model, d_state, d_conv, expand)

    def forward(self, values):
        return self.projection(values)


def test_mamba_baseline_uses_standard_model_interface(monkeypatch):
    monkeypatch.setattr(mamba_baseline, "Mamba", FakeMamba)
    configs = SimpleNamespace(
        task_name="classification", enc_in=3, num_class=4, dropout=0.1,
        d_model=8, d_state=12, d_conv=3, expand=2, e_layers=2,
    )
    model = mamba_baseline.Model(configs).eval()

    logits = model(torch.randn(2, 6, 3), torch.ones(2, 6), None, None)
    assert logits.shape == (2, 4)
    assert model.layers[0].settings == (8, 12, 3, 2)


def test_mamba_baseline_supports_patch_tokenizer(monkeypatch):
    monkeypatch.setattr(mamba_baseline, "Mamba", FakeMamba)
    configs = SimpleNamespace(
        task_name="classification", enc_in=3, num_class=4, dropout=0.1,
        d_model=8, d_state=6, d_conv=2, expand=1, e_layers=1,
        baseline_use_patch=True, baseline_patch_size=8,
    )
    model = mamba_baseline.Model(configs)
    assert model.uses_patch


def test_mamba_baseline_rejects_non_classification_tasks():
    configs = SimpleNamespace(task_name="long_term_forecast")
    try:
        mamba_baseline.Model(configs)
    except NotImplementedError as exc:
        assert "classification" in str(exc)
    else:
        raise AssertionError("MambaBaseline must be classification-only")

