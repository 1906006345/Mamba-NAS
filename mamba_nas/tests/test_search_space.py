import numpy as np

from mamba_nas.search_space import Genome, decode, encode, enumerate_space, space_size


def test_space_is_exactly_864_and_roundtrips():
    genomes = list(enumerate_space())
    assert space_size() == len(genomes) == 864
    assert len({genome.sha256 for genome in genomes}) == 864
    for genome in genomes:
        assert decode(encode(genome)) == genome
        assert genome.sha256 == genome.sha256


def test_invalid_gene_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        decode(np.asarray([99, 0, 0, 0, 0, 0, 0, 0]))


def test_scan_order_space_is_conditional_and_roundtrips():
    genomes = list(enumerate_space(use_scan_order=True))
    assert space_size(use_scan_order=True) == len(genomes) == 1440
    assert all(
        not (g.tokenizer == "point" and g.scan_order == "patch_interleaved")
        for g in genomes
    )
    for genome in genomes:
        assert decode(encode(genome, use_scan_order=True)) == genome


def test_legacy_hash_and_eight_gene_encoding_are_preserved():
    genome = Genome(
        tokenizer="point", num_blocks=1, direction="forward", d_model=64,
        d_state=8, d_conv=2, expand=1, pooling="mean",
    )
    assert len(encode(genome)) == 8
    assert decode(encode(genome)).scan_order == "time_major"
    assert "scan_order" not in genome.canonical_json()


def test_point_interleaved_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="requires tokenizer"):
        Genome(
            tokenizer="point", num_blocks=1, direction="forward", d_model=64,
            d_state=8, d_conv=2, expand=1, pooling="mean",
            scan_order="patch_interleaved",
        ).validate()

