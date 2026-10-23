from __future__ import annotations

import numpy as np

from .search_space import active_gene_names, bounds, decode, repair_vector


def uniform_crossover_arrays(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator, probability: float = 0.9,
    use_scan_order: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.asarray(parent_a, dtype=int).copy(), np.asarray(parent_b, dtype=int).copy()
    if rng.random() < probability:
        exchange = rng.random(len(first)) < 0.5
        original = first.copy()
        first[exchange] = second[exchange]
        second[exchange] = original[exchange]
    first = np.asarray(repair_vector(first, rng=rng), dtype=int)
    second = np.asarray(repair_vector(second, rng=rng), dtype=int)
    decode(first)
    decode(second)
    return first, second


def resample_mutation_array(
    vector: np.ndarray, rng: np.random.Generator, probability: float | None = None,
    use_scan_order: bool | None = None,
) -> np.ndarray:
    original = np.asarray(vector, dtype=int)
    result = original.copy()
    use_scan_order = len(result) == 9 if use_scan_order is None else use_scan_order
    names = active_gene_names(use_scan_order)
    probability = 1.0 / len(names) if probability is None else probability
    _, upper = bounds(use_scan_order)
    for gene, maximum in enumerate(upper):
        if rng.random() < probability:
            choices = [value for value in range(maximum + 1) if value != result[gene]]
            result[gene] = int(rng.choice(choices))
    result = np.asarray(repair_vector(result, original=original, rng=rng), dtype=int)
    decode(result)
    return result


def pymoo_operators(use_scan_order: bool = False):
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
                    X[0, mating, :], X[1, mating, :], rng, probability=1.0,
                    use_scan_order=use_scan_order,
                )
            return offspring

    class ResampleCategoricalMutation(Mutation):
        def _do(self, problem, X, **kwargs):
            rng = kwargs.get("random_state") or np.random.default_rng()
            return np.stack([
                resample_mutation_array(row, rng, use_scan_order=use_scan_order)
                for row in X
            ])

    return UniformCategoricalCrossover(), ResampleCategoricalMutation()


def pymoo_sampling(use_scan_order: bool = False):
    try:
        from pymoo.core.sampling import Sampling
    except ImportError as exc:
        raise RuntimeError("pymoo is required to run NSGA-II") from exc

    class LegalIntegerSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            rng = kwargs.get("random_state") or np.random.default_rng()
            lower, upper = bounds(use_scan_order)
            rows = []
            while len(rows) < n_samples:
                row = np.asarray(
                    [rng.integers(low, high + 1) for low, high in zip(lower, upper)],
                    dtype=int,
                )
                try:
                    decode(row)
                except ValueError:
                    continue
                rows.append(row)
            return np.stack(rows)

    return LegalIntegerSampling()
