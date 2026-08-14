from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
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
    synthetic_collection,
)
from .macs import estimate_macs, trainable_parameters
from .model import MambaTSCClassifier
from .operators import pymoo_operators
from .search_space import GENE_NAMES, bounds, decode, enumerate_space
from .training import seed_everything, train_classifier


def new_run_id(prefix: str = "nas") -> str:
    return f"{prefix}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _atomic_npz(path: Path, **arrays) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".part.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _save_torch(path: Path, value) -> None:
    import torch

    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    temporary.replace(path)


class CandidateEvaluator:
    def __init__(
        self,
        collection: SeriesCollection,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        dataset_dir: Path,
        config: SearchConfig,
        mixer_factory=None,
    ):
        self.collection = collection
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.dataset_dir = dataset_dir
        self.search_dir = ensure_dir(dataset_dir / "search")
        self.candidates_dir = ensure_dir(self.search_dir / "candidates")
        self.config = config
        self.mixer_factory = mixer_factory
        train_samples = [collection.samples[i] for i in train_indices]
        validation_samples = [collection.samples[i] for i in validation_indices]
        self.stats = fit_normalization(train_samples)
        self.train_samples = normalize_samples(train_samples, self.stats)
        self.validation_samples = normalize_samples(validation_samples, self.stats)
        self.train_labels = collection.labels[train_indices]
        self.validation_labels = collection.labels[validation_indices]
        self.max_objective_macs = max(
            math.log1p(
                estimate_macs(
                    genome,
                    collection.max_length,
                    collection.input_channels,
                    len(collection.class_names),
                ).total
            )
            for genome in enumerate_space()
        )
        self.cache = self._load_cache()
        _atomic_npz(
            self.search_dir / "normalization_stats.npz", mean=self.stats.mean, std=self.stats.std
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
            return self.cache[genome.sha256]
        last_error = None
        for attempt in (1, 2):
            try:
                row = self._train(genome, attempt)
                self.cache[genome.sha256] = row
                append_jsonl(self.search_dir / "evaluations.jsonl", row)
                self.write_evaluations_csv()
                return row
            except Exception as exc:
                last_error = exc
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
            "objective_f1": 1.0,
            "objective_macs": math.log1p(macs.total),
            "training_seconds": 0.0,
            "epochs_trained": 0,
            "status": "failed",
            "attempt": 2,
            "error": repr(last_error),
        }
        self.cache[genome.sha256] = row
        append_jsonl(self.search_dir / "evaluations.jsonl", row)
        self.write_evaluations_csv()
        return row

    def _train(self, genome, attempt: int) -> dict:
        import torch

        seed_everything(self.config.seed)
        train_loader = make_loader(
            self.train_samples, self.train_labels, self.config.batch_size, True, self.config.seed
        )
        validation_loader = make_loader(
            self.validation_samples,
            self.validation_labels,
            self.config.batch_size,
            False,
            self.config.seed,
        )
        model = MambaTSCClassifier(
            self.collection.input_channels,
            len(self.collection.class_names),
            genome,
            dropout=self.config.dropout,
            mixer_factory=self.mixer_factory,
        )
        parameters = trainable_parameters(model)
        macs = estimate_macs(
            genome,
            self.collection.max_length,
            self.collection.input_channels,
            len(self.collection.class_names),
        )
        result = train_classifier(
            model,
            train_loader,
            validation_loader,
            self.config.device,
            self.config.epochs,
            self.config.patience,
            self.config.learning_rate,
            self.config.weight_decay,
        )
        candidate_dir = ensure_dir(self.candidates_dir / genome.sha256)
        atomic_json(candidate_dir / "genome.json", genome.to_dict())
        atomic_csv(candidate_dir / "history.csv", result.history)
        _atomic_npz(
            candidate_dir / "val_predictions.npz",
            logits=result.logits,
            predictions=result.predictions,
            targets=result.targets,
        )
        _save_torch(candidate_dir / "checkpoint.pt", result.model.state_dict())
        row = {
            "candidate_hash": genome.sha256,
            **genome.to_dict(),
            **result.metrics,
            "parameters": parameters,
            "macs": macs.total,
            "objective_f1": 1.0 - result.metrics["macro_f1"],
            "objective_macs": math.log1p(macs.total),
            "training_seconds": result.training_seconds,
            "epochs_trained": len(result.history),
            "status": "completed",
            "attempt": attempt,
        }
        atomic_json(candidate_dir / "metrics.json", {**row, "mac_breakdown": macs.to_dict()})
        return row

    def write_evaluations_csv(self) -> None:
        rows = sorted(self.cache.values(), key=lambda row: row["candidate_hash"])
        atomic_csv(self.search_dir / "evaluations.csv", rows)


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
    return {"high_accuracy": high, "knee": knee, "low_cost": low}


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


def _generation_snapshot(search_dir: Path, generation: int, evaluator: CandidateEvaluator, algorithm) -> None:
    front = nondominated([row for row in evaluator.cache.values() if row["status"] == "completed"])
    generation_dir = ensure_dir(search_dir / "generations")
    rows = []
    if algorithm.pop is not None:
        for individual in algorithm.pop:
            genome = decode(individual.X)
            cached = evaluator.cache.get(genome.sha256, {})
            rows.append({"candidate_hash": genome.sha256, **genome.to_dict(), **cached})
    atomic_csv(generation_dir / f"generation_{generation:04d}.csv", rows)
    atomic_csv(generation_dir / f"generation_{generation:04d}_pareto.csv", front)
    atomic_json(
        generation_dir / f"generation_{generation:04d}.json",
        {
            "generation": generation,
            "unique_evaluations": len(evaluator.cache),
            "hypervolume": _hypervolume(front, evaluator.max_objective_macs),
            "hypervolume_reference": [1.1, evaluator.max_objective_macs * 1.01],
        },
    )


def run_search(
    dataset: str,
    config: SearchConfig,
    run_id: str | None = None,
    resume: bool = False,
    synthetic: bool = False,
    mixer_factory=None,
) -> Path:
    try:
        import torch
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
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
    state_path = search_dir / "search_state.pkl"
    collection = synthetic_collection(config.seed) if synthetic else load_train(config.data_dir, dataset)
    split_path = dataset_dir / "split_indices.npz"
    if resume and split_path.exists():
        saved = np.load(split_path)
        train_indices, validation_indices = saved["inner_train"], saved["validation"]
    else:
        train_indices, validation_indices = stratified_inner_split(collection.labels, config.seed)
        _atomic_npz(split_path, inner_train=train_indices, validation=validation_indices)
    manifest = {
        "project": "MambaTSC-NSGA2",
        "run_id": run_id,
        "dataset": dataset_name,
        "test_accessed": False,
        "split": {"strategy": "stratified_80_20", "seed": config.seed},
        "class_names": collection.class_names,
        "input_channels": collection.input_channels,
        "max_sequence_length": collection.max_length,
        "config": config.to_dict(),
    }
    atomic_json(dataset_dir / "manifest.json", manifest)
    evaluator = CandidateEvaluator(
        collection, train_indices, validation_indices, dataset_dir, config, mixer_factory=mixer_factory
    )
    lower, upper = bounds()

    class ArchitectureProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=len(GENE_NAMES), n_obj=2, xl=np.asarray(lower), xu=np.asarray(upper), vtype=int
            )
            self.evaluator = evaluator

        def _evaluate(self, x, out, *args, **kwargs):
            row = self.evaluator.evaluate(x)
            out["F"] = [row["objective_f1"], row["objective_macs"]]

    if resume and state_path.exists():
        import cloudpickle

        with state_path.open("rb") as handle:
            saved_state = cloudpickle.load(handle)
        algorithm = saved_state["algorithm"]
        algorithm.problem.evaluator = evaluator
        np.random.set_state(saved_state["numpy_random_state"])
        random.setstate(saved_state["python_random_state"])
        torch.set_rng_state(saved_state["torch_random_state"])
        if torch.cuda.is_available() and saved_state.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(saved_state["cuda_random_state"])
    else:
        crossover, mutation = pymoo_operators()
        algorithm = NSGA2(
            pop_size=config.population_size,
            sampling=IntegerRandomSampling(),
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
        )
        algorithm.setup(ArchitectureProblem(), seed=config.seed, verbose=True)

    while algorithm.has_next() and len(evaluator.cache) < config.max_unique_candidates:
        algorithm.next()
        generation = int(algorithm.n_gen)
        _generation_snapshot(search_dir, generation, evaluator, algorithm)
        active_evaluator = algorithm.problem.evaluator
        algorithm.problem.evaluator = None
        try:
            state = {
                "algorithm": algorithm,
                "numpy_random_state": np.random.get_state(),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            atomic_pickle(state_path, state)
            atomic_pickle(search_dir / "generations" / f"generation_{generation:04d}_state.pkl", state)
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
    atomic_json(search_dir / "representatives.json", select_representatives(front))
    manifest["search_completed"] = True
    manifest["unique_evaluations"] = len(evaluator.cache)
    manifest["pareto_size"] = len(front)
    atomic_json(dataset_dir / "manifest.json", manifest)
    return dataset_dir
