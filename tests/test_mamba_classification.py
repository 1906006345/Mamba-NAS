from types import SimpleNamespace

import torch

import models.Mamba as mamba_model


class FakeMamba(torch.nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.projection = torch.nn.Linear(d_model, d_model)

    def forward(self, values):
        return self.projection(values)


def _config(task_name):
    return SimpleNamespace(
        task_name=task_name,
        pred_len=12,
        enc_in=4,
        c_out=4,
        num_class=5,
        d_model=16,
        expand=2,
        d_ff=8,
        d_conv=2,
        e_layers=1,
        embed="timeF",
        freq="h",
        dropout=0.0,
    )


def test_mamba_classification_shape_and_padding_invariance(monkeypatch):
    monkeypatch.setattr(mamba_model, "Mamba", FakeMamba)
    model = mamba_model.Model(_config("classification")).eval()
    values = torch.randn(2, 48, 4)
    mask = torch.tensor([[True] * 48, [True] * 31 + [False] * 17])
    logits = model(values, mask, None, None)
    changed = values.clone()
    changed[1, 31:] = torch.randn_like(changed[1, 31:]) * 1000
    torch.testing.assert_close(logits, model(changed, mask, None, None))
    assert logits.shape == (2, 5)


def test_mamba_forecast_shape(monkeypatch):
    monkeypatch.setattr(mamba_model, "Mamba", FakeMamba)
    model = mamba_model.Model(_config("long_term_forecast")).eval()
    output = model(torch.randn(2, 48, 4), torch.randn(2, 48, 4), None, None)
    assert output.shape == (2, 12, 4)


def test_mamba_e_layers_builds_and_executes_all_layers(monkeypatch):
    monkeypatch.setattr(mamba_model, "Mamba", FakeMamba)
    config = _config("classification")
    config.e_layers = 3
    model = mamba_model.Model(config).eval()

    assert model.e_layers == 3
    assert len(model.mamba_layers) == 3
    assert len(model.extra_mamba_layers) == 2

    single = _config("classification")
    single.e_layers = 1
    single_model = mamba_model.Model(single)
    multi_parameters = sum(parameter.numel() for parameter in model.parameters())
    single_parameters = sum(parameter.numel() for parameter in single_model.parameters())
    one_layer_parameters = sum(parameter.numel() for parameter in single_model.mamba.parameters())
    assert multi_parameters == single_parameters + 2 * one_layer_parameters

    logits = model(torch.randn(2, 48, 4), torch.ones(2, 48), None, None)
    assert logits.shape == (2, 5)


def test_mamba_rejects_invalid_e_layers(monkeypatch):
    monkeypatch.setattr(mamba_model, "Mamba", FakeMamba)
    config = _config("classification")
    config.e_layers = 0
    try:
        mamba_model.Model(config)
    except ValueError as exc:
        assert "e_layers" in str(exc)
    else:
        raise AssertionError("Mamba must reject e_layers < 1")
