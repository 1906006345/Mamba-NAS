from __future__ import annotations

import csv
import copy
import datetime as dt
import json
import logging
import math
import random
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import append_jsonl, atomic_csv, atomic_json, atomic_pickle, ensure_dir, load_json
from .config import SearchConfig
from .data import (
    SeriesCollection,
    fit_normalization,
    load_train,
    make_loader,
    normalize_samples,
    stratified_inner_split,
    stratified_kfold_splits,
    synthetic_collection,
)
from .logging_utils import configure_logging
from .macs import estimate_macs, trainable_parameters
from .metrics import classification_metrics
from .model import MambaTSCClassifier
from .operators import pymoo_operators, pymoo_sampling, resample_mutation_array, uniform_crossover_arrays
from .search_space import GENE_NAMES, SCAN_GENE_NAMES, active_gene_names, bounds, decode, enumerate_space, space_size
from .search_space import CHOICES, Genome, encode, genome_from_mapping
from .training import seed_everything, train_classifier


LOGGER = logging.getLogger(__name__)


STRUCTURE_WEIGHTS = {
    "tokenizer": 2.0,
    "direction": 2.0,
    "num_blocks": 1.0,
    "d_model": 1.0,
    "pooling": 1.0,
    "scan_order": 2.0,
    "d_state": 0.5,
    "d_conv": 0.5,
    "expand": 0.5,
}
STRUCTURE_NICHES = tuple(
    (tokenizer, direction, capacity)
    for tokenizer in CHOICES["tokenizer"]
    for direction in CHOICES["direction"]
    for capacity in ("small", "medium", "large")
)


def capacity_level(genome: Genome) -> str:
    if genome.d_model == 64 and genome.num_blocks == 1:
        return "small"
    if genome.d_model == 128 and genome.num_blocks == 3:
        return "large"
    return "medium"


def structure_key(genome: Genome) -> tuple[str, str, str]:
    return (genome.tokenizer, genome.direction, capacity_level(genome))


def architecture_distance(first: Genome, second: Genome) -> float:
    return float(
        sum(
            STRUCTURE_WEIGHTS[name]
            for name in SCAN_GENE_NAMES
            if getattr(first, name) != getattr(second, name)
        )
    )


def _crowding_distance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    count, dimensions = values.shape if values.ndim == 2 else (0, 0)
    distances = np.zeros(count, dtype=float)
    if count <= 2:
        distances.fill(np.inf)
        return distances
    for dimension in range(dimensions):
        order = np.argsort(values[:, dimension], kind="mergesort")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = values[order[-1], dimension] - values[order[0], dimension]
        if span <= 0:
            continue
        for position in range(1, count - 1):
            index = order[position]
            if np.isfinite(distances[index]):
                distances[index] += (
                    values[order[position + 1], dimension]
                    - values[order[position - 1], dimension]
                ) / span
    return distances


def decision_crowding(genomes: list[Genome]) -> np.ndarray:
    count = len(genomes)
    if count <= 2:
        return np.full(count, np.inf, dtype=float)
    distances = np.asarray(
        [
            min(
                architecture_distance(genome, other)
                for other_index, other in enumerate(genomes)
                if other_index != index
            )
            for index, genome in enumerate(genomes)
        ],
        dtype=float,
    )
    return distances


def _normalize_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    result = np.zeros_like(values)
    low, high = values[finite].min(), values[finite].max()
    if high > low:
        result[finite] = (values[finite] - low) / (high - low)
    else:
        result[finite] = 1.0
    result[~finite] = 1.0
    return result


class DiversityArchive:
    def __init__(self, max_elites_per_niche: int = 3):
        self.max_elites_per_niche = max_elites_per_niche
        self.entries: dict[tuple[str, str, str], list[dict]] = {}
        self.generation = 0
        self.hashes: set[str] = set()

    def update(self, row: dict, generation: int) -> None:
        if row.get("status") != "completed":
            return
        if row["candidate_hash"] in self.hashes:
            return
        genome = genome_from_mapping(row)
        key = structure_key(genome)
        stored = dict(row)
        stored["structure_key"] = "|".join(key)
        stored["capacity_level"] = key[2]
        stored["archive_generation"] = generation
        candidates = self.entries.setdefault(key, [])
        candidates[:] = nondominated(candidates + [stored])
        if len(candidates) > self.max_elites_per_niche:
            objectives = np.asarray(
                [[item["objective_f1"], item["objective_macs"]] for item in candidates]
            )
            objective_scores = _normalize_finite(_crowding_distance(objectives))
            genomes = [genome_from_mapping(item) for item in candidates]
            decision_scores = _normalize_finite(decision_crowding(genomes))
            combined = np.where(
                np.isinf(objective_scores) | np.isinf(decision_scores),
                np.inf,
                0.7 * objective_scores + 0.3 * decision_scores,
            )
            keep = np.argsort(-combined, kind="mergesort")[: self.max_elites_per_niche]
            candidates[:] = [candidates[index] for index in keep]
        self.generation = max(self.generation, generation)
        self.hashes = {item["candidate_hash"] for item in self.rows()}

    def rows(self) -> list[dict]:
        return [row for rows in self.entries.values() for row in rows]

    def sample_rows(self, count: int, rng) -> list[dict]:
        rows = self.rows()
        if not rows:
            return []
        indices = rng.choice(len(rows), size=count, replace=len(rows) < count)
        return [rows[int(index)] for index in np.asarray(indices).reshape(-1)]

    def write_csv(self, path: Path) -> None:
        fields = [
            "candidate_hash", "structure_key", "tokenizer", "direction", "capacity_level",
            "macro_f1", "macs", "fold_macro_f1_std", "archive_generation",
        ]
        atomic_csv(path, self.rows(), fieldnames=fields)


class MacroMicroArchive(DiversityArchive):
    """Two-level archive: one macro representative plus local micro elites per niche."""

    def __init__(self, max_elites_per_niche: int = 3):
        super().__init__(max_elites_per_niche=max_elites_per_niche)
        self.macro_entries: dict[tuple[str, str, str], dict] = {}

    def update(self, row: dict, generation: int) -> None:
        super().update(row, generation)
        for key, candidates in self.entries.items():
            if not candidates:
                continue
            objectives = np.asarray(
                [[item["objective_f1"], item["objective_macs"]] for item in candidates]
            )
            objective_scores = _normalize_finite(_crowding_distance(objectives))
            genomes = [genome_from_mapping(item) for item in candidates]
            decision_scores = _normalize_finite(decision_crowding(genomes))
            scores = 0.7 * objective_scores + 0.3 * decision_scores
            index = int(np.argmax(scores))
            self.macro_entries[key] = candidates[index]

    def macro_rows(self) -> list[dict]:
        return list(self.macro_entries.values())

    def micro_rows(self) -> list[dict]:
        return self.rows()

    def write_macro_csv(self, path: Path) -> None:
        fields = [
            "candidate_hash", "structure_key", "tokenizer", "direction", "capacity_level",
            "macro_f1", "macs", "fold_macro_f1_std", "archive_generation",
        ]
        atomic_csv(path, self.macro_rows(), fieldnames=fields)

    def write_micro_csv(self, path: Path) -> None:
        self.write_csv(path)


EVALUATION_FIELDS = [
    "candidate_hash", *SCAN_GENE_NAMES, "macro_f1", "accuracy", "parameters", "macs",
    "flops", "macs_source", "flops_source",
    "objective_f1", "objective_macs", "training_seconds", "epochs_trained",
    "epochs_trained_mean", "validation_strategy", "cv_folds", "fold_macro_f1_mean",
    "fold_macro_f1_std", "fold_accuracy_mean", "fold_accuracy_std", "status", "attempt",
    "created_by", "parent_hash", "generation", "changed_gene", "archive_source",
    "fallback_stage", "neighbor_type", "parent_scan_order", "child_scan_order",
    "coupled_partner", "error",
]


def one_gene_neighbors(
    genome: Genome, use_scan_order: bool = False
) -> list[tuple[Genome, str]]:
    neighbors = []
    preferred = (
        ("scan_order",) if use_scan_order else ()
    ) + ("tokenizer", "direction", "pooling", "num_blocks", "d_model", "d_state", "d_conv", "expand")
    for name in preferred:
        choices = CHOICES[name]
        for value in choices:
            if value == getattr(genome, name):
                continue
            values = genome.to_dict()
            values[name] = value
            try:
                neighbors.append((Genome(**values).validate(), name))
            except ValueError:
                continue
    return neighbors


def coupled_scan_neighbors(genome: Genome) -> list[tuple[Genome, str, str]]:
    """Legal two-gene neighbors coupling scan_order with a related gene."""
    neighbors = []
    target_scan = (
        "patch_interleaved" if genome.scan_order == "time_major" else "time_major"
    )
    for partner in ("tokenizer", "direction", "d_conv"):
        for value in CHOICES[partner]:
            if value == getattr(genome, partner):
                continue
            values = genome.to_dict()
            values["scan_order"] = target_scan
            values[partner] = value
            try:
                child = Genome(**values).validate()
            except ValueError:
                continue
            neighbors.append((child, f"scan_order+{partner}", partner))
    return neighbors


def diversity_statistics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "archive_size": 0, "structure_coverage": 0.0, "tokenizer_coverage": 0.0,
            "direction_coverage": 0.0, "tokenizer_entropy": 0.0, "direction_entropy": 0.0,
            "scan_order_coverage": 0.0, "scan_order_entropy": 0.0,
            "time_major_pareto_count": 0, "patch_interleaved_pareto_count": 0,
            "mean_genome_distance": 0.0,
        }
    genomes = [genome_from_mapping(row) for row in rows]
    keys = {structure_key(genome) for genome in genomes}
    def entropy(values):
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log2(probabilities)).sum())
    pairwise = [
        architecture_distance(genomes[i], genomes[j])
        for i in range(len(genomes)) for j in range(i + 1, len(genomes))
    ]
    return {
        "archive_size": len(rows),
        "structure_coverage": len(keys) / len(STRUCTURE_NICHES),
        "tokenizer_coverage": len({genome.tokenizer for genome in genomes}) / len(CHOICES["tokenizer"]),
        "direction_coverage": len({genome.direction for genome in genomes}) / len(CHOICES["direction"]),
        "tokenizer_entropy": entropy([genome.tokenizer for genome in genomes]),
        "direction_entropy": entropy([genome.direction for genome in genomes]),
        "scan_order_coverage": len({genome.scan_order for genome in genomes}) / len(CHOICES["scan_order"]),
        "scan_order_entropy": entropy([genome.scan_order for genome in genomes]),
        "time_major_pareto_count": sum(genome.scan_order == "time_major" for genome in genomes),
        "patch_interleaved_pareto_count": sum(genome.scan_order == "patch_interleaved" for genome in genomes),
        "mean_genome_distance": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def _vector_hash(vector) -> str:
    return decode(vector).sha256


def _unique_vectors(vectors, cache: dict[str, dict], seen: set[str]) -> list[np.ndarray]:
    result = []
    for vector in vectors:
        array = np.asarray(vector, dtype=int)
        candidate_hash = _vector_hash(array)
        if candidate_hash in cache or candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        result.append(array)
    return result


def _random_unique_vectors(
    count: int, cache: dict[str, dict], seen: set[str], rng,
    use_scan_order: bool = False,
) -> list[np.ndarray]:
    lower, upper = bounds(use_scan_order)
    result = []
    attempts = 0
    while len(result) < count and attempts < max(100, count * 20):
        vector = np.asarray([rng.integers(low, high + 1) for low, high in zip(lower, upper)], dtype=int)
        attempts += 1
        try:
            candidate_hash = _vector_hash(vector)
        except ValueError:
            continue
        if candidate_hash in cache or candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        result.append(vector)
    return result


def _genome_rows_from_population(population, cache: dict[str, dict]) -> list[dict]:
    rows = []
    for individual in population:
        genome = decode(individual.X)
        row = cache.get(genome.sha256)
        if row is not None and row.get("status") == "completed":
            rows.append(row)
    return rows


def _local_search_vectors(
    evaluator: "CandidateEvaluator",
    population,
    archive: DiversityArchive | None,
    count: int,
    rng,
    generation: int,
    records: list[dict],
    seen: set[str],
) -> list[np.ndarray]:
    completed = [row for row in evaluator.cache.values() if row.get("status") == "completed"]
    seeds = nondominated(completed)
    if archive is not None:
        seeds.extend(archive.rows())
    seeds.extend(_genome_rows_from_population(population, evaluator.cache))
    unique_seed_hashes = set()
    vectors = []
    for row in seeds:
        genome = genome_from_mapping(row)
        if genome.sha256 in unique_seed_hashes:
            continue
        unique_seed_hashes.add(genome.sha256)
        vectors.append(genome)
    if vectors:
        order = rng.permutation(len(vectors))
        vectors = [vectors[int(i)] for i in order]
    result = []
    neighbor_limit = evaluator.config.local_search_max_neighbors
    for parent in vectors:
        neighbors = one_gene_neighbors(
            parent, evaluator.config.use_scan_order
        )[:neighbor_limit]
        for child, changed_gene in neighbors:
            child_hash = child.sha256
            if child_hash in evaluator.cache or child_hash in seen:
                continue
            seen.add(child_hash)
            result.append(np.asarray(
                encode(child, evaluator.config.use_scan_order), dtype=int
            ))
            parent_row = next(
                (row for row in seeds if row.get("candidate_hash") == parent.sha256),
                {},
            )
            records.append({
                "generation": generation,
                "parent_hash": parent.sha256,
                "child_hash": child_hash,
                "changed_gene": changed_gene,
                "parent_macro_f1": parent_row.get("macro_f1", ""),
                "parent_macs": parent_row.get("macs", ""),
                "neighbor_type": "single",
                "parent_scan_order": parent.scan_order,
                "child_scan_order": child.scan_order,
                "coupled_partner": "",
            })
            if len(result) >= count:
                return result
    return result


def _coupled_local_search_vectors(
    evaluator: "CandidateEvaluator",
    population,
    count: int,
    rng,
    generation: int,
    records: list[dict],
    seen: set[str],
) -> list[np.ndarray]:
    """Pareto multi-start search with single and scan-coupled neighborhoods."""
    completed = [row for row in evaluator.cache.values() if row.get("status") == "completed"]
    global_front = nondominated(completed)
    representatives = select_representatives(global_front) if global_front else {}
    ordered_rows = [
        representatives[role]
        for role in ("high_f1", "knee", "low_macs")
        if role in representatives
    ]
    representative_hashes = {row["candidate_hash"] for row in ordered_rows}
    ordered_rows.extend(
        row for row in global_front if row["candidate_hash"] not in representative_hashes
    )
    ordered_rows.extend(_genome_rows_from_population(population, evaluator.cache))
    seeds = []
    seed_hashes = set()
    for row in ordered_rows:
        if row["candidate_hash"] in seed_hashes:
            continue
        seed_hashes.add(row["candidate_hash"])
        seeds.append((row, genome_from_mapping(row)))

    # Allocate fractional quotas cumulatively. For example, five local slots
    # with ratio 0.30 alternate between one and two coupled children instead of
    # rounding to two every generation (which would silently become 40%).
    raw_coupled = count * evaluator.config.coupled_neighbor_ratio
    cumulative = math.floor(generation * raw_coupled + 0.5)
    previous = math.floor((generation - 1) * raw_coupled + 0.5)
    coupled_slots = min(count, max(0, cumulative - previous))
    single_slots = count - coupled_slots
    result: list[np.ndarray] = []
    positions: dict[tuple[str, str], int] = {}

    def add_round_robin(kind: str, target_total: int) -> None:
        while len(result) < target_total:
            progress = False
            for parent_row, parent in seeds:
                key = (parent.sha256, kind)
                position = positions.get(key, 0)
                if kind == "single":
                    candidates = [
                        (child, changed, "")
                        for child, changed in one_gene_neighbors(parent, True)[
                            : evaluator.config.local_search_max_neighbors
                        ]
                    ]
                else:
                    candidates = coupled_scan_neighbors(parent)
                while position < len(candidates):
                    child, changed_gene, partner = candidates[position]
                    position += 1
                    child_hash = child.sha256
                    if child_hash in evaluator.cache or child_hash in seen:
                        continue
                    seen.add(child_hash)
                    result.append(np.asarray(encode(child, use_scan_order=True), dtype=int))
                    records.append({
                        "generation": generation,
                        "parent_hash": parent.sha256,
                        "child_hash": child_hash,
                        "changed_gene": changed_gene,
                        "parent_macro_f1": parent_row.get("macro_f1", ""),
                        "parent_macs": parent_row.get("macs", ""),
                        "archive_source": (
                            "global_pareto"
                            if parent_row.get("candidate_hash") in {
                                row["candidate_hash"] for row in global_front
                            }
                            else "current_population"
                        ),
                        "neighbor_type": kind,
                        "parent_scan_order": parent.scan_order,
                        "child_scan_order": child.scan_order,
                        "coupled_partner": partner,
                    })
                    progress = True
                    break
                positions[key] = position
                if len(result) >= target_total:
                    break
            if not progress:
                break

    add_round_robin("single", single_slots)
    # Reserve the configured share for coupled moves even if the single quota
    # was under-filled, then use unused coupled slots for legal single moves.
    coupled_target = min(count, len(result) + coupled_slots)
    add_round_robin("coupled", coupled_target)
    if len(result) < count:
        add_round_robin("single", count)
    return result


def _local_search_vectors_v2(
    evaluator: "CandidateEvaluator",
    population,
    archive: MacroMicroArchive,
    count: int,
    rng,
    generation: int,
    records: list[dict],
    seen: set[str],
) -> list[np.ndarray]:
    """Layered multi-start local search for the macro/micro archive variant."""
    completed = [row for row in evaluator.cache.values() if row.get("status") == "completed"]
    global_front = nondominated(completed)
    representatives = select_representatives(global_front) if global_front else {}
    pareto_seeds = [
        (representatives[role], "global_pareto")
        for role in ("high_f1", "knee", "low_macs")
        if role in representatives
    ]

    macro_rows = archive.macro_rows()
    micro_rows = archive.micro_rows()
    macro_keys = {structure_key(genome_from_mapping(row)) for row, _ in pareto_seeds}
    archive_candidates = []
    archive_rows = [(row, "macro_archive") for row in macro_rows]
    archive_rows.extend((row, "micro_archive") for row in micro_rows)
    if archive_rows:
        order = rng.permutation(len(archive_rows))
        archive_rows = [archive_rows[int(index)] for index in order]
    for row, source in archive_rows:
        genome = genome_from_mapping(row)
        key = structure_key(genome)
        if key not in macro_keys:
            archive_candidates.append((row, source))
            macro_keys.add(key)
    population_rows = [
        (row, "current_population")
        for row in _genome_rows_from_population(population, evaluator.cache)
    ]

    def unique_pool(pool: list[tuple[dict, str]]) -> list[tuple[dict, str]]:
        output = []
        hashes = set()
        for row, source in pool:
            candidate_hash = row.get("candidate_hash")
            if candidate_hash in hashes:
                continue
            hashes.add(candidate_hash)
            output.append((row, source))
        if output:
            order = rng.permutation(len(output))
            output = [output[int(index)] for index in order]
        return output

    ratios = np.asarray([
        evaluator.config.pareto_seed_ratio,
        evaluator.config.archive_seed_ratio,
        evaluator.config.population_seed_ratio,
    ], dtype=float)
    raw_quotas = count * ratios
    quotas = np.floor(raw_quotas).astype(int)
    for index in np.argsort(-(raw_quotas - quotas), kind="mergesort")[: count - int(quotas.sum())]:
        quotas[index] += 1
    pools = [
        unique_pool(pareto_seeds),
        unique_pool(archive_candidates),
        unique_pool(population_rows),
    ]
    seeds: list[tuple[dict, str]] = []
    used_hashes: set[str] = set()
    used_keys: set[tuple[str, str, str]] = set()
    for pool, quota in zip(pools, quotas):
        selected = 0
        for row, source in pool:
            if len(seeds) >= count or selected >= int(quota):
                break
            genome = genome_from_mapping(row)
            key = structure_key(genome)
            if row.get("candidate_hash") in used_hashes or key in used_keys:
                continue
            used_hashes.add(row.get("candidate_hash"))
            used_keys.add(key)
            seeds.append((row, source))
            selected += 1
    # Fill any under-supplied quota from all seed pools, preserving uniqueness.
    all_candidates = unique_pool(pareto_seeds + archive_candidates + population_rows)
    for row, source in all_candidates:
        if len(seeds) >= count:
            break
        genome = genome_from_mapping(row)
        key = structure_key(genome)
        if row.get("candidate_hash") in used_hashes or key in used_keys:
            continue
        used_hashes.add(row.get("candidate_hash"))
        used_keys.add(key)
        seeds.append((row, source))

    result: list[np.ndarray] = []
    neighbor_limit = evaluator.config.local_search_max_neighbors

    def add_round_robin(seed_rows: list[tuple[dict, str]], fallback_stage: str) -> None:
        neighbor_sets = []
        for parent_row, source in seed_rows:
            parent = genome_from_mapping(parent_row)
            neighbor_sets.append(
                (parent_row, source, parent, one_gene_neighbors(parent)[:neighbor_limit], 0)
            )
        while len(result) < count:
            progress = False
            updated = []
            for parent_row, source, parent, neighbors, position in neighbor_sets:
                while position < len(neighbors):
                    child, changed_gene = neighbors[position]
                    position += 1
                    child_hash = child.sha256
                    if child_hash in evaluator.cache or child_hash in seen:
                        continue
                    seen.add(child_hash)
                    result.append(np.asarray(encode(child), dtype=int))
                    records.append({
                        "generation": generation,
                        "parent_hash": parent.sha256,
                        "child_hash": child_hash,
                        "changed_gene": changed_gene,
                        "parent_macro_f1": parent_row.get("macro_f1", ""),
                        "parent_macs": parent_row.get("macs", ""),
                        "archive_source": source,
                        "fallback_stage": fallback_stage,
                        "neighbor_type": "single",
                        "parent_scan_order": parent.scan_order,
                        "child_scan_order": child.scan_order,
                        "coupled_partner": "",
                    })
                    progress = True
                    break
                updated.append((parent_row, source, parent, neighbors, position))
                if len(result) >= count:
                    break
            neighbor_sets = updated
            if not progress:
                break

    # Strict phase: one seed per niche and quota-controlled multi-start expansion.
    add_round_robin(seeds, "")
    if len(result) < count:
        # Relaxed phase: permit a second seed from an already used niche before
        # surrendering the remaining local-search budget to normal variation.
        relaxed = [
            (row, source)
            for row, source in unique_pool(
                [(row, "global_pareto") for row in global_front]
                + [(row, "macro_archive") for row in macro_rows]
                + [(row, "micro_archive") for row in micro_rows]
                + population_rows
            )
            if row.get("candidate_hash") not in used_hashes
        ]
        add_round_robin(relaxed, "relaxed_seed")
    return result


def _sample_mixed_parents(population, archive: DiversityArchive | None, count: int, config: SearchConfig, rng):
    population_vectors = [np.asarray(individual.X, dtype=int) for individual in population]
    seen = set()
    result = []
    diversity_count = int(round(count * config.diversity_parent_ratio)) if archive is not None else 0
    random_count = int(round(count * config.random_immigrant_ratio))
    for row in archive.sample_rows(diversity_count, rng) if archive is not None else []:
        result.append(np.asarray(encode(genome_from_mapping(row)), dtype=int))
    result.extend(population_vectors[: max(0, count - diversity_count - random_count)])
    result.extend(_random_unique_vectors(random_count, {}, seen, rng))
    if len(result) < count:
        result.extend(population_vectors[: count - len(result)])
    return result[:count]


def _niche_architecture_crowding(genomes: list[Genome]) -> np.ndarray:
    """Decision-space crowding proxy used by the dual-space survival operator."""
    if len(genomes) <= 2:
        return np.full(len(genomes), np.inf, dtype=float)
    return decision_crowding(genomes)


def _make_dual_space_survival(objective_weight: float, decision_weight: float):
    from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
    from pymoo.operators.survival.rank_and_crowding.classes import get_crowding_function
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from pymoo.util.randomized_argsort import randomized_argsort

    class DualSpaceSurvival(RankAndCrowding):
        def __init__(self):
            super().__init__()
            self.nds = NonDominatedSorting()
            self.crowding_func = get_crowding_function("cd")
            self.objective_weight = float(objective_weight)
            self.decision_weight = float(decision_weight)

        def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
            F = pop.get("F").astype(float, copy=False)
            survivors = []
            fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
            for rank, front in enumerate(fronts):
                front = np.asarray(front, dtype=int)
                objective = self.crowding_func.do(F[front, :], n_remove=0)
                front_genomes = [decode(pop[index].X) for index in front]
                decision = _niche_architecture_crowding(front_genomes)
                objective_norm = _normalize_finite(objective)
                decision_norm = _normalize_finite(decision)
                combined = (
                    self.objective_weight * objective_norm
                    + self.decision_weight * decision_norm
                )
                if len(survivors) + len(front) > n_survive:
                    n_remove = len(survivors) + len(front) - n_survive
                    keep = randomized_argsort(
                        combined,
                        order="descending",
                        method="numpy",
                        random_state=random_state,
                    )[:-n_remove]
                else:
                    keep = np.arange(len(front))
                for position, index in enumerate(front):
                    pop[index].set("rank", rank)
                    pop[index].set("crowding", combined[position])
                survivors.extend(front[keep])
            return pop[survivors]

    return DualSpaceSurvival()


def _make_hybrid_infill(algorithm, pop, n_offsprings, evaluator, archive, config, generation, rng):
    """Create one generation of standard, archive, immigrant, and local-search children."""
    from pymoo.core.population import Population

    evaluator.generation = generation
    vectors = []
    seen: set[str] = set()
    local_records: list[dict] = []
    local_count = (
        int(np.ceil(n_offsprings * config.local_search_ratio))
        if config.use_pareto_local_search
        else 0
    )
    if local_count:
        if config.use_coupled_local_search:
            vectors.extend(
                _coupled_local_search_vectors(
                    evaluator,
                    pop,
                    local_count,
                    rng,
                    generation,
                    local_records,
                    seen,
                )
            )
        elif config.use_macro_micro_archive and isinstance(archive, MacroMicroArchive):
            vectors.extend(
                _local_search_vectors_v2(
                    evaluator,
                    pop,
                    archive,
                    local_count,
                    rng,
                    generation,
                    local_records,
                    seen,
                )
            )
        else:
            vectors.extend(
                _local_search_vectors(
                    evaluator,
                    pop,
                    archive,
                    local_count,
                    rng,
                    generation,
                    local_records,
                    seen,
                )
            )
        for record in local_records:
            evaluator.pending_metadata[record["child_hash"]] = {
                "created_by": "pareto_local_search",
                "parent_hash": record["parent_hash"],
                "generation": generation,
                "changed_gene": record["changed_gene"],
                "archive_source": record.get("archive_source", "global_pareto"),
                "fallback_stage": record.get("fallback_stage", ""),
                "neighbor_type": record.get("neighbor_type", "single"),
                "parent_scan_order": record.get("parent_scan_order", "time_major"),
                "child_scan_order": record.get("child_scan_order", "time_major"),
                "coupled_partner": record.get("coupled_partner", ""),
            }
            evaluator.local_search_metadata[record["child_hash"]] = dict(record)

    actual_local_children = len(local_records)
    single_local_children = sum(
        record.get("neighbor_type", "single") == "single" for record in local_records
    )
    coupled_local_children = sum(
        record.get("neighbor_type") == "coupled" for record in local_records
    )
    fallback_local_children = sum(
        bool(record.get("fallback_stage")) for record in local_records
    )
    fallback_variation_children = max(0, local_count - actual_local_children)
    evaluator.record_generation_budget({
        "generation": generation,
        "requested_local_slots": local_count,
        "actual_local_children": actual_local_children,
        "single_local_children": single_local_children,
        "coupled_local_children": coupled_local_children,
        "fallback_local_children": fallback_local_children,
        "fallback_variation_children": fallback_variation_children,
        "local_search_slot_fill_rate": (
            actual_local_children / local_count if local_count else 0.0
        ),
    })

    # The immigrant fraction is measured against the complete offspring
    # budget, so the configured 60/30/10 source mix remains reproducible even
    # when local-search slots are enabled alongside normal variation.
    remaining = n_offsprings - len(vectors)
    immigrant_count = min(
        remaining,
        int(np.floor(n_offsprings * config.random_immigrant_ratio)),
    )
    immigrants = _random_unique_vectors(
        immigrant_count, evaluator.cache, seen, rng, config.use_scan_order
    )
    for vector in immigrants:
            evaluator.pending_metadata[_vector_hash(vector)] = {
                "created_by": "random_immigrant",
                "generation": generation,
                "archive_source": "",
            }
    vectors.extend(immigrants)

    variation_count = n_offsprings - len(vectors)
    current_vectors = [np.asarray(individual.X, dtype=int) for individual in pop]
    archive_vectors = []
    if archive is not None and config.use_diversity_archive:
        for row in archive.sample_rows(max(2, variation_count * 2), rng):
            archive_vectors.append(
                np.asarray(encode(genome_from_mapping(row), config.use_scan_order), dtype=int)
            )
    parent_archive_probability = 0.0
    if archive_vectors and current_vectors:
        parent_archive_probability = min(
            1.0,
            config.diversity_parent_ratio
            / max(1e-12, 1.0 - config.random_immigrant_ratio),
        )
    parent_pool = current_vectors + archive_vectors
    if not parent_pool:
        parent_pool = _random_unique_vectors(
            2, evaluator.cache, seen, rng, config.use_scan_order
        )
    fallback_variation_remaining = fallback_variation_children
    while len(vectors) < n_offsprings and variation_count > 0:
        used_archive_parent = False
        def choose_parent():
            nonlocal used_archive_parent
            if archive_vectors and rng.random() < parent_archive_probability:
                used_archive_parent = True
                return archive_vectors[int(rng.integers(len(archive_vectors)))]
            return current_vectors[int(rng.integers(len(current_vectors)))]

        parent_a, parent_b = choose_parent(), choose_parent()
        child_a, child_b = uniform_crossover_arrays(
            parent_a, parent_b, rng, use_scan_order=config.use_scan_order
        )
        for child in (child_a, child_b):
            child = resample_mutation_array(
                child, rng, use_scan_order=config.use_scan_order
            )
            child_hash = _vector_hash(child)
            if child_hash in evaluator.cache or child_hash in seen:
                continue
            seen.add(child_hash)
            created_by = "diversity_archive" if used_archive_parent else "nsga_crossover"
            fallback_stage = ""
            if fallback_variation_remaining > 0:
                created_by = "fallback_variation"
                fallback_stage = "variation"
                fallback_variation_remaining -= 1
            evaluator.pending_metadata[child_hash] = {
                "created_by": created_by,
                "generation": generation,
                "archive_source": "micro_archive" if used_archive_parent else "",
                "fallback_stage": fallback_stage,
            }
            vectors.append(child)
            if len(vectors) >= n_offsprings:
                break
        # Avoid an infinite loop if the finite search space is exhausted.
        if len(seen) + len(evaluator.cache) >= space_size(config.use_scan_order):
            break
    if len(vectors) < n_offsprings:
        fillers = _random_unique_vectors(
            n_offsprings - len(vectors), evaluator.cache, seen, rng,
            config.use_scan_order,
        )
        for vector in fillers:
            candidate_hash = _vector_hash(vector)
            is_fallback = fallback_variation_remaining > 0
            evaluator.pending_metadata[candidate_hash] = {
                "created_by": "fallback_variation" if is_fallback else "random_immigrant",
                "generation": generation,
                "archive_source": "",
                "fallback_stage": "variation" if is_fallback else "",
            }
            if is_fallback:
                fallback_variation_remaining -= 1
        vectors.extend(fillers)
    if not vectors:
        raise RuntimeError("Hybrid infill could not create any unique offspring")
    return Population.new(X=np.asarray(vectors[:n_offsprings], dtype=int))


def _run_random_search(
    evaluator: "CandidateEvaluator", search_dir: Path, config: SearchConfig,
    state_path: Path, resume: bool = False,
) -> None:
    """Evaluate an independent, deterministic random Genome stream in batches.

    Random search deliberately does not construct a pymoo algorithm.  Batching by
    ``population_size`` gives it the same generation/evaluation checkpoints as
    evolutionary methods while retaining an independent candidate sequence.
    """
    rng = np.random.default_rng(config.seed)
    generation = 0
    if resume and state_path.exists():
        import cloudpickle

        with state_path.open("rb") as handle:
            saved = cloudpickle.load(handle)
        if saved.get("search_algorithm") != "random":
            raise ValueError("Saved search state does not belong to random search")
        rng.bit_generator.state = saved["rng_state"]
        generation = int(saved.get("generation", 0))
    while len(evaluator.cache) < config.max_unique_candidates:
        remaining = config.max_unique_candidates - len(evaluator.cache)
        count = min(config.population_size, remaining)
        vectors = _random_unique_vectors(
            count, evaluator.cache, set(), rng, config.use_scan_order
        )
        if not vectors:
            break
        evaluator.generation = generation + 1
        for vector in vectors:
            candidate_hash = _vector_hash(vector)
            evaluator.pending_metadata[candidate_hash] = {
                "created_by": "random_search",
                "generation": generation + 1,
            }
            evaluator.evaluate(vector)
        generation += 1
        snapshot = _generation_snapshot(search_dir, generation, evaluator, None)
        LOGGER.info(
            "[搜索][随机代完成] generation=%d evaluated=%d hypervolume=%.6f",
            generation,
            len(evaluator.cache),
            snapshot["hypervolume"],
        )
        try:
            atomic_pickle(
                state_path,
                {
                    "search_algorithm": "random",
                    "generation": generation,
                    "rng_state": rng.bit_generator.state,
                },
            )
        except OSError as exc:
            LOGGER.warning("[搜索][状态保存跳过] algorithm=random error=%r", exc)


def new_run_id(prefix: str = "nas") -> str:
    return f"{prefix}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _atomic_npz(path: Path, **arrays) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".part.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _save_cv_splits(path: Path, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    arrays = {"fold_count": np.asarray(len(splits), dtype=np.int64)}
    for fold_index, (train, validation) in enumerate(splits):
        arrays[f"fold_{fold_index:02d}_train"] = train
        arrays[f"fold_{fold_index:02d}_validation"] = validation
    _atomic_npz(path, **arrays)


def _load_cv_splits(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    with np.load(path) as saved:
        fold_count = int(saved["fold_count"])
        return [
            (
                saved[f"fold_{fold_index:02d}_train"].copy(),
                saved[f"fold_{fold_index:02d}_validation"].copy(),
            )
            for fold_index in range(fold_count)
        ]


def _validate_resume_protocol(dataset_dir: Path, config: SearchConfig, resume: bool) -> None:
    manifest_path = dataset_dir / "manifest.json"
    evaluations_path = dataset_dir / "search" / "evaluations.jsonl"
    if not manifest_path.exists():
        return
    previous = load_json(manifest_path).get("config", {})
    previous_strategy = previous.get("validation_strategy", "holdout")
    previous_folds = int(previous.get("cv_folds", 5))
    previous_seed = int(previous.get("seed", 2021))
    current = (
        config.validation_strategy,
        config.cv_folds if config.validation_strategy == "kfold" else None,
        config.seed,
        config.search_algorithm,
        config.search_variant,
        config.use_decision_crowding,
        config.use_diversity_archive,
        config.use_pareto_local_search,
        config.objective_weight,
        config.decision_weight,
        config.use_macro_micro_archive,
        config.use_scan_order,
        config.use_coupled_local_search,
        config.hypervolume_space,
        config.local_search_ratio,
        config.coupled_neighbor_ratio,
        config.archive_seed_ratio,
        config.pareto_seed_ratio,
        config.population_seed_ratio,
    )
    prior = (
        previous_strategy,
        previous_folds if previous_strategy == "kfold" else None,
        previous_seed,
        previous.get("search_algorithm", "nsga2"),
        previous.get("search_variant", "baseline"),
        bool(previous.get("use_decision_crowding", False)),
        bool(previous.get("use_diversity_archive", False)),
        bool(previous.get("use_pareto_local_search", False)),
        float(previous.get("objective_weight", 0.70)),
        float(previous.get("decision_weight", 0.30)),
        bool(previous.get("use_macro_micro_archive", False)),
        bool(previous.get("use_scan_order", False)),
        bool(previous.get("use_coupled_local_search", False)),
        previous.get("hypervolume_space", "active"),
        float(previous.get("local_search_ratio", 0.30)),
        float(previous.get("coupled_neighbor_ratio", 0.30)),
        float(previous.get("archive_seed_ratio", 0.30)),
        float(previous.get("pareto_seed_ratio", 0.50)),
        float(previous.get("population_seed_ratio", 0.20)),
    )
    if resume and current != prior:
        raise ValueError(
            "Cannot resume with a different validation protocol: "
            f"saved={prior}, requested={current}. Use a new --run-id."
        )
    if not resume and evaluations_path.exists():
        raise FileExistsError(
            f"{dataset_dir} already contains candidate evaluations. Use --resume with the same "
            "validation protocol or choose a new --run-id."
        )


class CandidateEvaluator:
    def __init__(
        self,
        collection: SeriesCollection,
        splits: list[tuple[np.ndarray, np.ndarray]],
        dataset_dir: Path,
        config: SearchConfig,
        mixer_factory=None,
    ):
        self.collection = collection
        self.splits = splits
        self.dataset_dir = dataset_dir
        self.search_dir = ensure_dir(dataset_dir / "search")
        self.candidates_dir = ensure_dir(self.search_dir / "candidates")
        self.config = config
        self.started_at = time.perf_counter()
        self.mixer_factory = mixer_factory
        self.pending_metadata: dict[str, dict] = {}
        self.local_search_metadata: dict[str, dict] = {}
        self.generation = 0
        self.local_search_records: list[dict] = []
        local_search_path = self.search_dir / "local_search.csv"
        if local_search_path.exists():
            with local_search_path.open("r", encoding="utf-8", newline="") as handle:
                self.local_search_records = list(csv.DictReader(handle))
        self.generation_budget_records: list[dict] = []
        generation_budget_path = self.search_dir / "generation_search_budget.csv"
        if generation_budget_path.exists():
            with generation_budget_path.open("r", encoding="utf-8", newline="") as handle:
                self.generation_budget_records = list(csv.DictReader(handle))
        self.fold_stats = []
        for fold_index, (train_indices, _) in enumerate(splits):
            train_samples = [collection.samples[i] for i in train_indices]
            stats = fit_normalization(train_samples)
            self.fold_stats.append(stats)
            if len(splits) == 1:
                stats_path = self.search_dir / "normalization_stats.npz"
            else:
                stats_path = self.search_dir / "folds" / f"fold_{fold_index:02d}" / "normalization_stats.npz"
            _atomic_npz(stats_path, mean=stats.mean, std=stats.std)
        if len(splits) > 1:
            validation_indices = np.concatenate([validation for _, validation in splits])
            if len(validation_indices) != len(collection.labels) or not np.array_equal(
                np.sort(validation_indices), np.arange(len(collection.labels))
            ):
                raise ValueError(
                    "K-fold validation indices must cover every TRAIN sample exactly once"
                )
        self.max_objective_macs = max(
            math.log1p(
                estimate_macs(
                    genome,
                    collection.max_length,
                    collection.input_channels,
                    len(collection.class_names),
                ).total
            )
            for genome in enumerate_space(
                config.use_scan_order or config.hypervolume_space == "scan_order"
            )
        )
        self.cache = self._load_cache()
        LOGGER.info(
            "[搜索][评估器就绪] dataset=%s strategy=%s folds=%d "
            "fold_sizes=%s cached_completed=%d",
            dataset_dir.name,
            config.validation_strategy,
            len(splits),
            [(len(train), len(validation)) for train, validation in splits],
            len(self.cache),
        )

    def _load_cache(self) -> dict[str, dict]:
        cache = {}
        path = self.search_dir / "evaluations.jsonl"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        # Completed candidates are reusable across restarts. Failed
                        # candidates are cached only for this process so a repaired
                        # CUDA/backend environment can retry them on the next run.
                        if row.get("status") == "completed":
                            cache[row["candidate_hash"]] = row
        return cache

    def evaluate(self, vector) -> dict:
        genome = decode(vector)
        if genome.sha256 in self.cache:
            LOGGER.info("[搜索][缓存命中] candidate=%s", genome.sha256[:12])
            return self.cache[genome.sha256]
        last_error = None
        for attempt in (1, 2):
            LOGGER.info(
                "[搜索][候选开始] index=%d candidate=%s attempt=%d genome=%s",
                len(self.cache) + 1,
                genome.sha256[:12],
                attempt,
                json.dumps(genome.to_dict(), sort_keys=True),
            )
            try:
                row = self._train(genome, attempt)
                row.update(self.pending_metadata.pop(genome.sha256, {}))
                row.setdefault("generation", self.generation)
                row.setdefault("created_by", "initial_sampling")
                self.cache[genome.sha256] = row
                append_jsonl(self.search_dir / "evaluations.jsonl", row)
                self.write_evaluations_csv()
                self._record_local_result(row)
                LOGGER.info(
                    "[搜索][候选完成] candidate=%s macro_f1=%.6f accuracy=%.6f "
                    "parameters=%d macs=%d epochs=%d seconds=%.2f",
                    genome.sha256[:12],
                    row["macro_f1"],
                    row["accuracy"],
                    row["parameters"],
                    row["macs"],
                    row["epochs_trained"],
                    row["training_seconds"],
                )
                return row
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "[搜索][候选失败] candidate=%s attempt=%d error=%r",
                    genome.sha256[:12],
                    attempt,
                    exc,
                )
                failure = {
                    "candidate_hash": genome.sha256,
                    "genome": genome.to_dict(),
                    "status": "failed",
                    "attempt": attempt,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(self.search_dir / "failures.jsonl", failure)
        macs = estimate_macs(
            genome,
            self.collection.max_length,
            self.collection.input_channels,
            len(self.collection.class_names),
        )
        row = {
            "candidate_hash": genome.sha256,
            **genome.to_dict(),
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "parameters": "",
            "macs": macs.total,
            "flops": macs.total * 2,
            "macs_source": "analytical_mamba_cost",
            "flops_source": "analytical_flops_2x_macs",
            "objective_f1": 1.0,
            "objective_macs": math.log1p(macs.total),
            "training_seconds": 0.0,
            "epochs_trained": 0,
            "validation_strategy": self.config.validation_strategy,
            "cv_folds": len(self.splits),
            "status": "failed",
            "attempt": 2,
            "error": repr(last_error),
        }
        row.update(self.pending_metadata.pop(genome.sha256, {}))
        row.setdefault("generation", self.generation)
        row.setdefault("created_by", "initial_sampling")
        self.cache[genome.sha256] = row
        append_jsonl(self.search_dir / "evaluations.jsonl", row)
        self.write_evaluations_csv()
        self._record_local_result(row)
        LOGGER.error(
            "[搜索][候选最终失败] candidate=%s attempts=2 error=%r",
            genome.sha256[:12],
            last_error,
        )
        return row

    def _record_local_result(self, row: dict) -> None:
        metadata = self.local_search_metadata.pop(row["candidate_hash"], None)
        if metadata is None:
            return
        self.local_search_records.append({**metadata, "child_macro_f1": row.get("macro_f1", ""), "child_macs": row.get("macs", ""), "child_status": row.get("status", "")})
        atomic_csv(self.search_dir / "local_search.csv", self.local_search_records)

    def record_generation_budget(self, row: dict) -> None:
        generation = str(row["generation"])
        self.generation_budget_records = [
            item for item in self.generation_budget_records
            if str(item.get("generation")) != generation
        ]
        self.generation_budget_records.append(dict(row))
        self.generation_budget_records.sort(key=lambda item: int(item["generation"]))
        atomic_csv(
            self.search_dir / "generation_search_budget.csv",
            self.generation_budget_records,
        )

    def _train(self, genome, attempt: int) -> dict:
        macs = estimate_macs(
            genome,
            self.collection.max_length,
            self.collection.input_channels,
            len(self.collection.class_names),
        )
        candidate_dir = ensure_dir(self.candidates_dir / genome.sha256)
        atomic_json(candidate_dir / "genome.json", genome.to_dict())
        histories, logits_all, predictions_all, targets_all, indices_all = [], [], [], [], []
        fold_metrics, training_seconds, epochs_total = [], 0.0, 0
        parameters = None
        for fold_index, ((train_indices, validation_indices), stats) in enumerate(
            zip(self.splits, self.fold_stats)
        ):
            fold_seed = self.config.seed + fold_index
            seed_everything(fold_seed)
            train_samples = normalize_samples(
                [self.collection.samples[i] for i in train_indices], stats
            )
            validation_samples = normalize_samples(
                [self.collection.samples[i] for i in validation_indices], stats
            )
            train_loader = make_loader(
                train_samples,
                self.collection.labels[train_indices],
                self.config.batch_size,
                True,
                fold_seed,
            )
            validation_loader = make_loader(
                validation_samples,
                self.collection.labels[validation_indices],
                self.config.batch_size,
                False,
                fold_seed,
            )
            model = MambaTSCClassifier(
                self.collection.input_channels,
                len(self.collection.class_names),
                genome,
                dropout=self.config.dropout,
                mixer_factory=self.mixer_factory,
            )
            if parameters is None:
                parameters = trainable_parameters(model)
            result = train_classifier(
                model,
                train_loader,
                validation_loader,
                self.config.device,
                self.config.epochs,
                self.config.patience,
                self.config.learning_rate,
                self.config.weight_decay,
                log_context=(
                    f"candidate={genome.sha256[:12]} attempt={attempt} "
                    f"fold={fold_index + 1}/{len(self.splits)}"
                ),
            )
            fold_dir = (
                candidate_dir
                if len(self.splits) == 1
                else ensure_dir(candidate_dir / "folds" / f"fold_{fold_index:02d}")
            )
            fold_history = (
                result.history
                if len(self.splits) == 1
                else [{"fold": fold_index, **row} for row in result.history]
            )
            atomic_csv(fold_dir / "history.csv", fold_history)
            _atomic_npz(
                fold_dir / "val_predictions.npz",
                logits=result.logits,
                predictions=result.predictions,
                targets=result.targets,
                sample_indices=validation_indices,
            )
            fold_row = {
                "fold": fold_index,
                "seed": fold_seed,
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                **result.metrics,
                "epochs_trained": len(result.history),
                "training_seconds": result.training_seconds,
            }
            atomic_json(fold_dir / "metrics.json", fold_row)
            histories.extend(fold_history)
            logits_all.append(result.logits)
            predictions_all.append(result.predictions)
            targets_all.append(result.targets)
            indices_all.append(validation_indices)
            fold_metrics.append(fold_row)
            training_seconds += result.training_seconds
            epochs_total += len(result.history)
            LOGGER.info(
                "[搜索][Fold完成] candidate=%s fold=%d/%d macro_f1=%.6f accuracy=%.6f",
                genome.sha256[:12],
                fold_index + 1,
                len(self.splits),
                result.metrics["macro_f1"],
                result.metrics["accuracy"],
            )

        logits = np.concatenate(logits_all)
        predictions = np.concatenate(predictions_all)
        targets = np.concatenate(targets_all)
        sample_indices = np.concatenate(indices_all)
        metrics = classification_metrics(targets, predictions)
        f1_values = np.asarray([row["macro_f1"] for row in fold_metrics], dtype=float)
        accuracy_values = np.asarray([row["accuracy"] for row in fold_metrics], dtype=float)
        if len(self.splits) > 1:
            atomic_csv(candidate_dir / "history.csv", histories)
        _atomic_npz(
            candidate_dir / "val_predictions.npz",
            logits=logits,
            predictions=predictions,
            targets=targets,
            sample_indices=sample_indices,
        )
        row = {
            "candidate_hash": genome.sha256,
            **genome.to_dict(),
            **metrics,
            "parameters": parameters,
            "macs": macs.total,
            "flops": macs.total * 2,
            "macs_source": "analytical_mamba_cost",
            "flops_source": "analytical_flops_2x_macs",
            "objective_f1": 1.0 - metrics["macro_f1"],
            "objective_macs": math.log1p(macs.total),
            "training_seconds": training_seconds,
            "epochs_trained": epochs_total,
            "epochs_trained_mean": epochs_total / len(self.splits),
            "validation_strategy": self.config.validation_strategy,
            "cv_folds": len(self.splits),
            "fold_macro_f1_mean": float(f1_values.mean()),
            "fold_macro_f1_std": (
                float(f1_values.std(ddof=1)) if len(f1_values) > 1 else 0.0
            ),
            "fold_accuracy_mean": float(accuracy_values.mean()),
            "fold_accuracy_std": (
                float(accuracy_values.std(ddof=1)) if len(accuracy_values) > 1 else 0.0
            ),
            "status": "completed",
            "attempt": attempt,
        }
        atomic_json(
            candidate_dir / "metrics.json",
            {**row, "mac_breakdown": macs.to_dict(), "fold_metrics": fold_metrics},
        )
        return row

    def write_evaluations_csv(self) -> None:
        rows = sorted(self.cache.values(), key=lambda row: row["candidate_hash"])
        fields = [*EVALUATION_FIELDS]
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        atomic_csv(self.search_dir / "evaluations.csv", rows, fieldnames=fields)


def nondominated(rows: list[dict]) -> list[dict]:
    result = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["objective_f1"] <= candidate["objective_f1"]
            and other["objective_macs"] <= candidate["objective_macs"]
            and (
                other["objective_f1"] < candidate["objective_f1"]
                or other["objective_macs"] < candidate["objective_macs"]
            )
            for other in rows
        )
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda row: (row["macs"], -row["macro_f1"]))


def select_representatives(front: list[dict]) -> dict[str, dict]:
    if not front:
        raise ValueError("Cannot select representatives from an empty front")
    high = max(front, key=lambda row: (row["macro_f1"], -row["macs"]))
    low = min(front, key=lambda row: (row["macs"], -row["macro_f1"]))
    objectives = np.asarray([[row["objective_f1"], row["objective_macs"]] for row in front])
    ranges = np.ptp(objectives, axis=0)
    normalized = (objectives - objectives.min(axis=0)) / np.where(ranges > 0, ranges, 1)
    knee = front[int(np.linalg.norm(normalized, axis=1).argmin())]
    return {"high_f1": high, "knee": knee, "low_macs": low}


def abort_all_failed(evaluator: CandidateEvaluator, dataset_dir: Path, manifest: dict) -> None:
    failure_path = evaluator.search_dir / "failures.jsonl"
    failures = []
    if failure_path.exists():
        with failure_path.open("r", encoding="utf-8") as handle:
            failures = [json.loads(line) for line in handle if line.strip()]
    counts = Counter(row.get("error", "unknown error") for row in failures)
    common_error, common_count = counts.most_common(1)[0] if counts else ("unknown error", 0)
    summary = {
        "failed_candidates": sum(row["status"] == "failed" for row in evaluator.cache.values()),
        "failure_records": len(failures),
        "most_common_error": common_error,
        "most_common_error_count": common_count,
        "latest_traceback": failures[-1].get("traceback") if failures else None,
    }
    atomic_json(evaluator.search_dir / "failure_summary.json", summary)
    manifest["search_completed"] = False
    manifest["failure_summary"] = summary
    atomic_json(dataset_dir / "manifest.json", manifest)
    LOGGER.error("[搜索][中止] dataset=%s summary=%s", dataset_dir.name, summary)
    raise RuntimeError(
        f"All evaluated candidates failed. Most common error ({common_count} records): "
        f"{common_error}. See {evaluator.search_dir / 'failure_summary.json'} and "
        f"{failure_path}. Fix the backend, then rerun; failed candidates are retryable."
    )


def _hypervolume(front: list[dict], max_objective_macs: float) -> float:
    if not front:
        return 0.0
    values = np.asarray([[row["objective_f1"], row["objective_macs"]] for row in front])
    ordered = values[np.argsort(values[:, 0])]
    reference = np.asarray([1.1, max_objective_macs * 1.01])
    area, previous_y = 0.0, reference[1]
    for x, y in ordered:
        if y < previous_y:
            area += (reference[0] - x) * (previous_y - y)
            previous_y = y
    return float(area)


def _generation_snapshot(search_dir: Path, generation: int, evaluator: CandidateEvaluator, algorithm) -> dict:
    from .convergence_metrics import hv_auc

    front = nondominated([row for row in evaluator.cache.values() if row["status"] == "completed"])
    generation_dir = ensure_dir(search_dir / "generations")
    rows = []
    if algorithm is not None and algorithm.pop is not None:
        for individual in algorithm.pop:
            genome = decode(individual.X)
            cached = evaluator.cache.get(genome.sha256, {})
            rows.append({"candidate_hash": genome.sha256, **genome.to_dict(), **cached})
    atomic_csv(generation_dir / f"generation_{generation:04d}.csv", rows)
    atomic_csv(generation_dir / f"generation_{generation:04d}_pareto.csv", front)
    metadata = {
        "search_algorithm": evaluator.config.search_algorithm,
        "search_variant": evaluator.config.search_variant,
        "generation": generation,
        "unique_evaluations": len(evaluator.cache),
        "elapsed_seconds": time.perf_counter() - evaluator.started_at,
        "training_seconds": float(sum(float(row.get("training_seconds", 0.0) or 0.0) for row in evaluator.cache.values())),
        "hypervolume": _hypervolume(front, evaluator.max_objective_macs),
        "hypervolume_reference": [1.1, evaluator.max_objective_macs * 1.01],
        **diversity_statistics(front),
    }
    archive = getattr(algorithm, "diversity_archive", None)
    if archive is not None:
        archive_stats = diversity_statistics(archive.rows())
        metadata.update({f"elite_{key}": value for key, value in archive_stats.items()})
        if isinstance(archive, MacroMicroArchive):
            macro_stats = diversity_statistics(archive.macro_rows())
            micro_stats = diversity_statistics(archive.micro_rows())
            metadata.update({f"macro_{key}": value for key, value in macro_stats.items()})
            metadata.update({f"micro_{key}": value for key, value in micro_stats.items()})
    front_hashes = {row["candidate_hash"] for row in front}
    archive_source_rows = [
        row for row in evaluator.cache.values()
        if row.get("archive_source") in {"macro_archive", "micro_archive"}
    ]
    archive_source_pareto = [
        row for row in archive_source_rows if row.get("candidate_hash") in front_hashes
    ]
    metadata.update({
        "archive_source_children": len(archive_source_rows),
        "archive_source_pareto_count": len(archive_source_pareto),
        "archive_source_pareto_rate": (
            len(archive_source_pareto) / len(archive_source_rows)
            if archive_source_rows else 0.0
        ),
    })
    local_rows = [
        row for row in evaluator.local_search_records
        if str(row.get("generation", generation)) == str(generation)
    ]
    if local_rows:
        positive, deltas, pareto_hits, contributions = [], [], [], []
        front_hashes = {row["candidate_hash"] for row in front}
        for row in local_rows:
            try:
                parent_f1 = float(row.get("parent_macro_f1", ""))
                child_f1 = float(row.get("child_macro_f1", ""))
                delta = child_f1 - parent_f1
                deltas.append(delta)
                positive.append(delta > 0)
            except (TypeError, ValueError):
                pass
            child_hash = row.get("child_hash", "")
            pareto_hits.append(child_hash in front_hashes)
            if child_hash in front_hashes:
                child_row = next((item for item in front if item["candidate_hash"] == child_hash), None)
                if child_row is not None:
                    without_child = [item for item in front if item["candidate_hash"] != child_hash]
                    contributions.append(
                        _hypervolume(front, evaluator.max_objective_macs)
                        - _hypervolume(without_child, evaluator.max_objective_macs)
                    )
        metadata.update({
            "local_search_children": len(local_rows),
            "local_search_positive_f1_rate": float(np.mean(positive)) if positive else 0.0,
            "local_search_pareto_entry_rate": float(np.mean(pareto_hits)) if pareto_hits else 0.0,
            "local_search_mean_hv_contribution": float(np.mean(contributions)) if contributions else 0.0,
            "local_search_mean_delta_f1": float(np.mean(deltas)) if deltas else 0.0,
        })
    else:
        metadata.update({
            "local_search_children": 0,
            "local_search_positive_f1_rate": 0.0,
            "local_search_pareto_entry_rate": 0.0,
            "local_search_mean_hv_contribution": 0.0,
            "local_search_mean_delta_f1": 0.0,
        })
    def local_type_metrics(rows: list[dict], neighbor_type: str) -> dict:
        selected = [
            row for row in rows
            if row.get("neighbor_type", "single") == neighbor_type
        ]
        deltas, hits, contributions = [], [], []
        for row in selected:
            try:
                deltas.append(
                    float(row.get("child_macro_f1", ""))
                    - float(row.get("parent_macro_f1", ""))
                )
            except (TypeError, ValueError):
                pass
            child_hash = row.get("child_hash", "")
            hit = child_hash in front_hashes
            hits.append(hit)
            if hit:
                without_child = [
                    item for item in front if item["candidate_hash"] != child_hash
                ]
                contributions.append(
                    _hypervolume(front, evaluator.max_objective_macs)
                    - _hypervolume(without_child, evaluator.max_objective_macs)
                )
        return {
            f"{neighbor_type}_local_children": len(selected),
            f"{neighbor_type}_pareto_entry_rate": float(np.mean(hits)) if hits else 0.0,
            f"{neighbor_type}_mean_delta_f1": float(np.mean(deltas)) if deltas else 0.0,
            f"{neighbor_type}_mean_hv_contribution": (
                float(np.mean(contributions)) if contributions else 0.0
            ),
        }
    metadata.update(local_type_metrics(local_rows, "single"))
    metadata.update(local_type_metrics(local_rows, "coupled"))
    budget_row = next(
        (
            row for row in evaluator.generation_budget_records
            if str(row.get("generation")) == str(generation)
        ),
        {},
    )
    for key in (
        "requested_local_slots",
        "actual_local_children",
        "single_local_children",
        "coupled_local_children",
        "fallback_local_children",
        "fallback_variation_children",
        "local_search_slot_fill_rate",
    ):
        value = budget_row.get(key, 0)
        metadata[key] = float(value) if key == "local_search_slot_fill_rate" else int(float(value))

    cumulative_rows = list(evaluator.local_search_records)
    cumulative_positive, cumulative_deltas, cumulative_hits, cumulative_contributions = [], [], [], []
    for row in cumulative_rows:
        try:
            parent_f1 = float(row.get("parent_macro_f1", ""))
            child_f1 = float(row.get("child_macro_f1", ""))
            delta = child_f1 - parent_f1
            cumulative_deltas.append(delta)
            cumulative_positive.append(delta > 0)
        except (TypeError, ValueError):
            pass
        child_hash = row.get("child_hash", "")
        hit = child_hash in front_hashes
        cumulative_hits.append(hit)
        if hit:
            without_child = [item for item in front if item["candidate_hash"] != child_hash]
            cumulative_contributions.append(
                _hypervolume(front, evaluator.max_objective_macs)
                - _hypervolume(without_child, evaluator.max_objective_macs)
            )
    cumulative_budgets = evaluator.generation_budget_records
    cumulative_requested = sum(int(float(row.get("requested_local_slots", 0))) for row in cumulative_budgets)
    cumulative_actual = sum(int(float(row.get("actual_local_children", 0))) for row in cumulative_budgets)
    metadata.update({
        "local_search_cumulative_children": len(cumulative_rows),
        "local_search_cumulative_positive_f1_rate": (
            float(np.mean(cumulative_positive)) if cumulative_positive else 0.0
        ),
        "local_search_cumulative_pareto_entry_rate": (
            float(np.mean(cumulative_hits)) if cumulative_hits else 0.0
        ),
        "local_search_cumulative_mean_hv_contribution": (
            float(np.mean(cumulative_contributions)) if cumulative_contributions else 0.0
        ),
        "local_search_cumulative_mean_delta_f1": (
            float(np.mean(cumulative_deltas)) if cumulative_deltas else 0.0
        ),
        "cumulative_requested_local_slots": cumulative_requested,
        "cumulative_actual_local_children": cumulative_actual,
        "cumulative_fallback_local_children": sum(
            int(float(row.get("fallback_local_children", 0))) for row in cumulative_budgets
        ),
        "cumulative_fallback_variation_children": sum(
            int(float(row.get("fallback_variation_children", 0))) for row in cumulative_budgets
        ),
        "cumulative_local_search_slot_fill_rate": (
            cumulative_actual / cumulative_requested if cumulative_requested else 0.0
        ),
    })
    atomic_json(generation_dir / f"generation_{generation:04d}.json", metadata)
    diversity_path = search_dir / "generation_diversity.csv"
    previous = []
    if diversity_path.exists():
        with diversity_path.open("r", encoding="utf-8", newline="") as handle:
            previous = [row for row in csv.DictReader(handle) if int(row["generation"]) != generation]
    all_rows = [*previous, metadata]
    all_rows.sort(key=lambda row: int(float(row.get("generation", 0))))
    metadata["hypervolume_auc"] = hv_auc(
        [float(row.get("unique_evaluations", 0)) for row in all_rows],
        [float(row.get("hypervolume", 0.0)) for row in all_rows],
        evaluator.config.max_unique_candidates,
    )
    reference_path = search_dir / "reference_front.csv"
    reference = None
    if reference_path.exists():
        try:
            reference_rows = list(csv.DictReader(reference_path.open("r", encoding="utf-8", newline="")))
            reference = np.asarray(
                [[float(row["objective_f1"]), float(row["objective_macs"])] for row in reference_rows],
                dtype=float,
            )
        except (OSError, KeyError, TypeError, ValueError):
            reference = None
    if reference is not None and len(reference):
        from .convergence_metrics import front_metrics

        metadata.update(front_metrics(
            np.asarray([[row["objective_f1"], row["objective_macs"]] for row in front], dtype=float),
            reference,
        ))
    else:
        metadata.update({
            "gd": float("nan"), "igd": float("nan"), "igd_plus": float("nan"),
            "epsilon_indicator": float("nan"), "spacing": float("nan"), "spread": float("nan"),
        })
    metadata.update({
        "best_macro_f1": max((float(row.get("macro_f1", 0.0)) for row in front), default=0.0),
        "low_macs": min((float(row.get("macs", 0.0)) for row in front), default=0.0),
    })
    atomic_json(generation_dir / f"generation_{generation:04d}.json", metadata)
    atomic_csv(diversity_path, all_rows)
    convergence_fields = [
        "search_algorithm", "search_variant", "generation", "unique_evaluations",
        "elapsed_seconds", "training_seconds", "hypervolume", "hypervolume_auc",
        "gd", "igd", "igd_plus", "epsilon_indicator", "spacing", "spread", "pareto_size",
        "best_macro_f1", "low_macs",
    ]
    convergence_path = search_dir / "convergence_metrics.csv"
    convergence_rows = []
    if convergence_path.exists():
        with convergence_path.open("r", encoding="utf-8", newline="") as handle:
            convergence_rows = [row for row in csv.DictReader(handle) if int(float(row["generation"])) != generation]
    atomic_csv(
        convergence_path,
        [*convergence_rows, {key: metadata.get(key, "") for key in convergence_fields}],
        fieldnames=convergence_fields,
    )
    return metadata


def run_search(
    dataset: str,
    config: SearchConfig,
    run_id: str | None = None,
    resume: bool = False,
    synthetic: bool = False,
    mixer_factory=None,
) -> Path:
    config.validate()
    try:
        import torch
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.decomposition.tchebicheff import Tchebicheff
    except ImportError as exc:
        raise RuntimeError("PyTorch and pymoo are required to run a search") from exc

    if mixer_factory is None:
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Official Mamba search requires a GPU-visible Linux container."
            )
        try:
            from mamba_ssm import Mamba as _OfficialMamba  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "mamba_ssm is not installed. Install mamba-ssm==2.1.0 after PyTorch, then run "
                "`python -m mamba_nas.cli verify-environment` before search."
            ) from exc

    run_id = run_id or new_run_id()
    dataset_name = "SyntheticUEA" if synthetic else dataset
    dataset_dir = ensure_dir(Path(config.output_dir) / run_id / dataset_name)
    search_dir = ensure_dir(dataset_dir / "search")
    _validate_resume_protocol(dataset_dir, config, resume)
    configure_logging(search_dir / "search.log")
    LOGGER.info(
        "[搜索][开始] run_id=%s dataset=%s algorithm=%s budget=%s population=%d max_candidates=%d "
        "epochs=%d device=%s resume=%s validation_strategy=%s cv_folds=%d",
        run_id,
        dataset_name,
        config.search_algorithm,
        config.budget,
        config.population_size,
        config.max_unique_candidates,
        config.epochs,
        config.device,
        resume,
        config.validation_strategy,
        config.cv_folds,
    )
    LOGGER.info(
        "[搜索][方法配置] variant=%s decision_crowding=%s objective_weight=%.3f "
        "decision_weight=%.3f diversity_archive=%s pareto_local_search=%s",
        config.search_variant,
        config.use_decision_crowding,
        config.objective_weight,
        config.decision_weight,
        config.use_diversity_archive,
        config.use_pareto_local_search,
    )
    state_path = search_dir / "search_state.pkl"
    collection = synthetic_collection(config.seed) if synthetic else load_train(config.data_dir, dataset)
    LOGGER.info(
        "[搜索][数据加载] dataset=%s samples=%d classes=%d channels=%d max_length=%d source=%s",
        dataset_name,
        len(collection.samples),
        len(collection.class_names),
        collection.input_channels,
        collection.max_length,
        "synthetic" if synthetic else "TRAIN.ts only",
    )
    if config.validation_strategy == "holdout":
        split_path = dataset_dir / "split_indices.npz"
        if resume and split_path.exists():
            with np.load(split_path) as saved:
                splits = [(saved["inner_train"].copy(), saved["validation"].copy())]
            split_source = "restored"
        else:
            train_indices, validation_indices = stratified_inner_split(
                collection.labels, config.seed
            )
            splits = [(train_indices, validation_indices)]
            _atomic_npz(
                split_path, inner_train=train_indices, validation=validation_indices
            )
            split_source = "created"
        split_manifest = {
            "strategy": "stratified_80_20",
            "seed": config.seed,
            "folds": 1,
            "path": split_path.name,
        }
    else:
        split_path = dataset_dir / "cv_indices.npz"
        if resume and split_path.exists():
            splits = _load_cv_splits(split_path)
            if len(splits) != config.cv_folds:
                raise ValueError(
                    f"Saved fold count {len(splits)} does not match cv_folds={config.cv_folds}"
                )
            split_source = "restored"
        else:
            splits = stratified_kfold_splits(
                collection.labels, config.cv_folds, config.seed
            )
            _save_cv_splits(split_path, splits)
            split_source = "created"
        split_manifest = {
            "strategy": "stratified_kfold",
            "seed": config.seed,
            "folds": config.cv_folds,
            "path": split_path.name,
        }
    LOGGER.info(
        "[搜索][数据划分] strategy=%s source=%s folds=%d fold_sizes=%s seed=%d",
        config.validation_strategy,
        split_source,
        len(splits),
        [(len(train), len(validation)) for train, validation in splits],
        config.seed,
    )
    manifest = {
        "project": "MambaTSC-NSGA2",
        "run_id": run_id,
        "dataset": dataset_name,
        "test_accessed": False,
        "split": split_manifest,
        "class_names": collection.class_names,
        "input_channels": collection.input_channels,
        "max_sequence_length": collection.max_length,
        "config": config.to_dict(),
    }
    atomic_json(dataset_dir / "manifest.json", manifest)
    evaluator = CandidateEvaluator(
        collection, splits, dataset_dir, config, mixer_factory=mixer_factory
    )
    archive = (
        MacroMicroArchive(max_elites_per_niche=3)
        if config.use_macro_micro_archive
        else DiversityArchive(max_elites_per_niche=3)
        if config.use_diversity_archive
        else None
    )
    if archive is not None:
        for row in evaluator.cache.values():
            archive.update(row, generation=0)
        archive.write_csv(search_dir / "diversity_archive.csv")
        if isinstance(archive, MacroMicroArchive):
            archive.write_macro_csv(search_dir / "macro_archive.csv")
            archive.write_micro_csv(search_dir / "micro_archive.csv")
    atomic_json(
        search_dir / "variant.json",
        {
            "search_algorithm": config.search_algorithm,
            "search_variant": config.search_variant,
            "use_decision_crowding": config.use_decision_crowding,
            "use_diversity_archive": config.use_diversity_archive,
            "use_pareto_local_search": config.use_pareto_local_search,
            "use_macro_micro_archive": config.use_macro_micro_archive,
            "use_scan_order": config.use_scan_order,
            "use_coupled_local_search": config.use_coupled_local_search,
            "hypervolume_space": config.hypervolume_space,
            "objective_weight": config.objective_weight,
            "decision_weight": config.decision_weight,
            "diversity_parent_ratio": config.diversity_parent_ratio,
            "random_immigrant_ratio": config.random_immigrant_ratio,
            "local_search_ratio": config.local_search_ratio,
            "local_search_max_neighbors": config.local_search_max_neighbors,
            "coupled_neighbor_ratio": config.coupled_neighbor_ratio,
            "archive_seed_ratio": config.archive_seed_ratio,
            "pareto_seed_ratio": config.pareto_seed_ratio,
            "population_seed_ratio": config.population_seed_ratio,
        },
    )
    active_names = active_gene_names(config.use_scan_order)
    lower, upper = bounds(config.use_scan_order)

    class ArchitectureProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=len(active_names), n_obj=2, xl=np.asarray(lower), xu=np.asarray(upper), vtype=int
            )
            self.evaluator = evaluator

        def _evaluate(self, x, out, *args, **kwargs):
            row = self.evaluator.evaluate(x)
            out["F"] = [row["objective_f1"], row["objective_macs"]]

    if config.search_algorithm == "random":
        if resume and state_path.exists():
            LOGGER.info("[搜索][恢复] random search uses deterministic seed stream; state=%s", state_path)
        _run_random_search(evaluator, search_dir, config, state_path, resume=resume)
        algorithm = None
    else:
        algorithm = None

    if config.search_algorithm != "random" and resume and state_path.exists():
        import cloudpickle

        with state_path.open("rb") as handle:
            saved_state = cloudpickle.load(handle)
        algorithm = saved_state["algorithm"]
        algorithm.problem.evaluator = evaluator
        algorithm.diversity_archive = archive
        algorithm.nas_evaluator = evaluator
        algorithm.nas_config = config
        np.random.set_state(saved_state["numpy_random_state"])
        random.setstate(saved_state["python_random_state"])
        torch.set_rng_state(saved_state["torch_random_state"])
        if torch.cuda.is_available() and saved_state.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(saved_state["cuda_random_state"])
        LOGGER.info(
            "[搜索][恢复] generation=%s cached_completed=%d state=%s",
            algorithm.n_gen,
            len(evaluator.cache),
            state_path,
        )
    elif config.search_algorithm != "random":
        crossover, mutation = pymoo_operators(config.use_scan_order)
        algorithm_kwargs = {
            "pop_size": config.population_size,
            "sampling": pymoo_sampling(config.use_scan_order),
            "crossover": crossover,
            "mutation": mutation,
            "eliminate_duplicates": True,
        }
        if config.use_decision_crowding:
            algorithm_kwargs["survival"] = _make_dual_space_survival(
                config.objective_weight,
                config.decision_weight,
            )
        if config.search_algorithm == "moead":
            effective_population = min(config.population_size, config.max_unique_candidates)
            ref_dirs = get_reference_directions(
                "uniform", 2, n_partitions=max(1, effective_population - 1)
            )
            algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=min(15, max(1, effective_population - 1)),
                decomposition=Tchebicheff(),
                sampling=algorithm_kwargs["sampling"],
                crossover=algorithm_kwargs["crossover"],
                mutation=algorithm_kwargs["mutation"],
            )
        else:
            algorithm = NSGA2(**algorithm_kwargs)
        algorithm.diversity_archive = archive
        algorithm.nas_evaluator = evaluator
        algorithm.nas_config = config
        algorithm.setup(ArchitectureProblem(), seed=config.seed, verbose=True)
        LOGGER.info(
            "[搜索][初始化] 新建 %s 种群，population=%d",
            config.search_algorithm.upper() if config.search_algorithm != "moead" else "MOEA/D",
            config.population_size,
        )

    while algorithm is not None and algorithm.has_next() and len(evaluator.cache) < config.max_unique_candidates:
        remaining_budget = config.max_unique_candidates - len(evaluator.cache)
        # All NSGA-II variants use the same unique-offspring infill protocol.
        # The archive/local-search mechanisms only change the source mix inside
        # ``_make_hybrid_infill``; baseline therefore has identical batch size,
        # duplicate handling, immigrant budget and generation checkpoints.
        if config.search_algorithm == "nsga2":
            offspring_count = min(int(algorithm.n_offsprings), remaining_budget)
            next_generation = int(
                math.ceil((len(evaluator.cache) + offspring_count) / config.population_size)
            )
            infills = _make_hybrid_infill(
                algorithm,
                algorithm.pop,
                offspring_count,
                evaluator,
                archive,
                config,
                next_generation,
                np.random.default_rng(config.seed + int(algorithm.n_iter + 1)),
            ) if algorithm.is_initialized else None
            if not algorithm.is_initialized:
                # Keep the initial sampling within the declared unique
                # candidate budget as well (important for tiny smoke runs).
                algorithm.pop_size = min(int(algorithm.pop_size), remaining_budget)
                algorithm.n_offsprings = min(int(algorithm.n_offsprings), remaining_budget)
                algorithm.next()
            else:
                algorithm.evaluator.eval(algorithm.problem, infills, algorithm=algorithm)
                algorithm.advance(infills=infills)
        else:
            algorithm.pop_size = min(int(algorithm.pop_size), remaining_budget)
            algorithm.n_offsprings = min(int(algorithm.n_offsprings), remaining_budget)
            algorithm.next()
        # MOEA/D in pymoo is loop-wise: one ``next`` call may evaluate one
        # neighbourhood child after initialization. Use the cumulative
        # evaluation count as its checkpoint index so convergence curves do
        # not overwrite multiple intermediate points with the same n_gen.
        generation = int(
            math.ceil(len(evaluator.cache) / config.population_size)
        )
        if archive is not None:
            for row in evaluator.cache.values():
                archive.update(row, generation)
            archive.write_csv(search_dir / "diversity_archive.csv")
            if isinstance(archive, MacroMicroArchive):
                archive.write_macro_csv(search_dir / "macro_archive.csv")
                archive.write_micro_csv(search_dir / "micro_archive.csv")
        snapshot = _generation_snapshot(search_dir, generation, evaluator, algorithm)
        completed = [row for row in evaluator.cache.values() if row["status"] == "completed"]
        LOGGER.info(
            "[搜索][代完成] generation=%d evaluated=%d completed=%d pareto=%d hypervolume=%.6f",
            generation,
            len(evaluator.cache),
            len(completed),
            len(nondominated(completed)),
            snapshot["hypervolume"],
        )
        active_evaluator = algorithm.problem.evaluator
        algorithm.problem.evaluator = None
        try:
            # pymoo's MOEA/D keeps a live generator while advancing one
            # neighbourhood member at a time. Generators cannot be pickled.
            # Serialize a shallow checkpoint copy with the transient generator
            # cleared; the active algorithm continues with its live generator.
            checkpoint_algorithm = algorithm
            if config.search_algorithm == "moead" and hasattr(algorithm, "generator"):
                checkpoint_algorithm = copy.copy(algorithm)
                checkpoint_algorithm.generator = None
                checkpoint_algorithm.state = None
            state = {
                "algorithm": checkpoint_algorithm,
                "numpy_random_state": np.random.get_state(),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            try:
                atomic_pickle(state_path, state)
            except (TypeError, AttributeError, OSError) as exc:
                LOGGER.warning("[搜索][状态保存跳过] algorithm=%s error=%r", config.search_algorithm, exc)
        finally:
            algorithm.problem.evaluator = active_evaluator
        completed = [row for row in evaluator.cache.values() if row["status"] == "completed"]
        if len(evaluator.cache) >= config.population_size and not completed:
            abort_all_failed(evaluator, dataset_dir, manifest)
        if len(evaluator.cache) >= config.max_unique_candidates:
            break
    evaluator.write_evaluations_csv()
    front = nondominated([row for row in evaluator.cache.values() if row["status"] == "completed"])
    if not front:
        abort_all_failed(evaluator, dataset_dir, manifest)
    atomic_csv(search_dir / "pareto_front.csv", front)
    representatives = select_representatives(front)
    atomic_json(search_dir / "representatives.json", representatives)
    manifest["search_completed"] = True
    manifest["unique_evaluations"] = len(evaluator.cache)
    manifest["pareto_size"] = len(front)
    atomic_json(dataset_dir / "manifest.json", manifest)
    LOGGER.info(
        "[搜索][完成] run_id=%s dataset=%s evaluated=%d pareto=%d output=%s",
        run_id,
        dataset_name,
        len(evaluator.cache),
        len(front),
        dataset_dir,
    )
    for role, row in representatives.items():
        LOGGER.info(
            "[搜索][代表架构] role=%s candidate=%s macro_f1=%.6f macs=%d",
            role,
            row["candidate_hash"][:12],
            row["macro_f1"],
            row["macs"],
        )
    return dataset_dir
