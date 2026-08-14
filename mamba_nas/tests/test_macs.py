from mamba_nas.macs import estimate_macs
from mamba_nas.search_space import Genome


def g(**changes):
    values = dict(
        tokenizer="point", num_blocks=1, direction="forward", d_model=64, d_state=8,
        d_conv=2, expand=1, pooling="mean"
    )
    values.update(changes)
    return Genome(**values)


def test_bidirectional_doubles_only_mamba_scan_component():
    forward = estimate_macs(g(), 128, 4, 5)
    bidirectional = estimate_macs(g(direction="bidirectional"), 128, 4, 5)
    assert bidirectional.mamba == 2 * forward.mamba
    assert bidirectional.tokenizer == forward.tokenizer
    assert bidirectional.classifier == forward.classifier


def test_patch_reduces_cost_and_capacity_increases_it():
    baseline = estimate_macs(g(), 128, 4, 5).total
    assert estimate_macs(g(tokenizer="patch8"), 128, 4, 5).total < baseline
    assert estimate_macs(g(num_blocks=3), 128, 4, 5).total > baseline
    assert estimate_macs(g(d_model=128), 128, 4, 5).total > baseline
    assert estimate_macs(g(expand=2), 128, 4, 5).total > baseline

