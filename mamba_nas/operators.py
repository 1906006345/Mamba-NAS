from __future__ import annotations

import numpy as np

from .search_space import GENE_NAMES, bounds, decode


def uniform_crossover_arrays(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator, probability: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.asarray(parent_a, dtype=int).copy(), np.asarray(parent_b, dtype=int).copy()
    if rng.random() < probability:
        exchange = rng.random(len(first)) < 0.5
        original = first.copy()
        first[exchange] = second[exchange]
        second[exchange] = original[exchange]
    decode(first)
    decode(second)
    return first, second


def resample_mutation_array(
    vector: np.ndarray, rng: np.random.Generator, probability: float | None = None
) -> np.ndarray:
    result = np.asarray(vector, dtype=int).copy()
    probability = 1.0 / len(GENE_NAMES) if probability is None else probability
    _, upper = bounds()
    for gene, maximum in enumerate(upper):
        if rng.random() < probability:
            choices = [value for value in range(maximum + 1) if value != result[gene]]
            result[gene] = int(rng.choice(choices))
    decode(result)
    return result


def pymoo_operators():
    try:
        from pymoo.core.crossover import Crossover
        from pymoo.core.mutation import Mutation
    except ImportError as exc:
        raise RuntimeError("pymoo is required to run NSGA-II") from exc

    class UniformCategoricalCrossover(Crossover):
        def __init__(self):
            super().__init__(2, 2, prob=0.9)

        def _do(self, problem, X, **kwargs):
            rng = kwargs.get("random_state") or np.random.default_rng()
            offspring = np.empty_like(X, dtype=int)
            # pymoo uses [n_parents, n_matings, n_genes].
            for mating in range(X.shape[1]):
                offspring[0, mating, :], offspring[1, mating, :] = uniform_crossover_arrays(
                    X[0, mating, :], X[1, mating, :], rng, probability=1.0
                )
            return offspring

    class ResampleCategoricalMutation(Mutation):
        def _do(self, problem, X, **kwargs):
            rng = kwargs.get("random_state") or np.random.default_rng()
            return np.stack([resample_mutation_array(row, rng) for row in X])

    return UniformCategoricalCrossover(), ResampleCategoricalMutation()
