import numpy as np

from mamba_nas.search_space import decode, encode, enumerate_space, space_size


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

