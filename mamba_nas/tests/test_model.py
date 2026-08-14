import itertools

import pytest
import torch

from mamba_nas.macs import trainable_parameters
from mamba_nas.model import MambaTSCClassifier
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
