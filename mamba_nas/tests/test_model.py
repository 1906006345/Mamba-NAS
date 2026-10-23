import itertools

import pytest
import torch

from mamba_nas.macs import trainable_parameters
from mamba_nas.model import MambaTSCClassifier, Tokenizer
from mamba_nas.search_space import Genome
from mamba_nas.tests.conftest import fake_mixer_factory


def genome(**changes):
    values = dict(
        tokenizer="point",
        num_blocks=1,
        direction="forward",
        d_model=64,
        d_state=8,
        d_conv=2,
        expand=1,
        pooling="mean",
    )
    values.update(changes)
    return Genome(**values)


@pytest.mark.parametrize(
    "tokenizer,direction,pooling,blocks",
    list(itertools.product(
        ("point", "patch8", "patch16"),
        ("forward", "bidirectional"),
        ("mean", "max", "meanmax"),
        (1, 2, 3),
    )),
)
def test_all_structural_choices_produce_logits(tokenizer, direction, pooling, blocks):
    model = MambaTSCClassifier(
        3,
        5,
        genome(tokenizer=tokenizer, direction=direction, pooling=pooling, num_blocks=blocks),
        dropout=0,
        mixer_factory=fake_mixer_factory,
    ).eval()
    values = torch.randn(2, 35, 3)
    mask = torch.tensor([[True] * 35, [True] * 23 + [False] * 12])
    assert model(values, mask).shape == (2, 5)


@pytest.mark.parametrize("tokenizer", ("point", "patch8", "patch16"))
def test_padding_values_cannot_change_logits(tokenizer):
    model = MambaTSCClassifier(
        2,
        3,
        genome(tokenizer=tokenizer, direction="bidirectional", pooling="meanmax", num_blocks=2),
        dropout=0,
        mixer_factory=fake_mixer_factory,
    ).eval()
    first = torch.randn(2, 31, 2)
    mask = torch.tensor([[True] * 19 + [False] * 12, [True] * 31])
    second = first.clone()
    second[0, 19:] = torch.randn_like(second[0, 19:]) * 1000
    torch.testing.assert_close(model(first, mask), model(second, mask), atol=1e-5, rtol=1e-5)


def test_shared_bidirectional_has_same_parameter_count():
    forward = MambaTSCClassifier(2, 3, genome(direction="forward"), mixer_factory=fake_mixer_factory)
    bidirectional = MambaTSCClassifier(
        2, 3, genome(direction="bidirectional"), mixer_factory=fake_mixer_factory
    )
    assert trainable_parameters(forward) == trainable_parameters(bidirectional)


def test_patch_interleaved_orders_patch_before_channel():
    tokenizer = Tokenizer(2, 1, "patch8", "patch_interleaved")
    with torch.no_grad():
        tokenizer.projection.weight.fill_(1.0)
        tokenizer.projection.bias.zero_()
    values = torch.stack((torch.arange(1, 17), torch.arange(101, 117)), dim=-1).float()
    tokens, token_mask = tokenizer(values.unsqueeze(0), torch.ones(1, 16, dtype=torch.bool))
    assert tokens.shape == (1, 4, 1)
    torch.testing.assert_close(
        tokens[0, :, 0], torch.tensor([36.0, 836.0, 100.0, 900.0])
    )
    assert token_mask.tolist() == [[True, True, True, True]]


def test_patch_interleaved_padding_values_cannot_change_logits():
    model = MambaTSCClassifier(
        3, 4, genome(tokenizer="patch8", scan_order="patch_interleaved"),
        dropout=0, mixer_factory=fake_mixer_factory,
    ).eval()
    first = torch.randn(2, 24, 3)
    mask = torch.tensor([[True] * 13 + [False] * 11, [True] * 24])
    second = first.clone()
    second[0, 13:] = torch.randn_like(second[0, 13:]) * 1000
    torch.testing.assert_close(model(first, mask), model(second, mask), atol=1e-5, rtol=1e-5)
