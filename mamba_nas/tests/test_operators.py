import numpy as np

from mamba_nas.operators import resample_mutation_array, uniform_crossover_arrays
from mamba_nas.search_space import decode


def test_operators_always_return_legal_categorical_genes():
    rng = np.random.default_rng(2021)
    first = np.zeros(8, dtype=int)
    second = np.asarray([2, 2, 1, 1, 1, 1, 1, 2])
    for _ in range(100):
        child_a, child_b = uniform_crossover_arrays(first, second, rng)
        decode(child_a)
        decode(child_b)
        decode(resample_mutation_array(child_a, rng))


def test_forced_mutation_changes_every_gene():
    original = np.zeros(8, dtype=int)
    mutated = resample_mutation_array(original, np.random.default_rng(7), probability=1.0)
    assert np.all(mutated != original)
    decode(mutated)


def test_pymoo_crossover_tensor_layout():
    crossover, mutation = __import__("mamba_nas.operators", fromlist=["pymoo_operators"]).pymoo_operators()
    parents = np.asarray(
        [
            [[0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 1, 1, 1, 1, 1, 2]],
            [[2, 2, 1, 1, 1, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0]],
        ]
    )
    children = crossover._do(None, parents, random_state=np.random.default_rng(4))
    assert children.shape == parents.shape
    for child in children.reshape(-1, 8):
        decode(child)
