from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np

from .artifacts import atomic_csv, atomic_json, ensure_dir, load_json
from .data import fit_normalization, load_test_for_refit, load_train, make_loader, normalize_samples
from .macs import estimate_macs, trainable_parameters
from .model import MambaTSCClassifier
from .profiling import profile_cuda
from .search_space import Genome
from .training import seed_everything, train_full_classifier


FINAL_SEEDS = (2021, 2022, 2023)


def _materialize_alias(source: Path, destination: Path, role: str, genome: Genome) -> None:
    for name in (
        "checkpoint.pt",
        "history.csv",
        "test_predictions.npz",
        "metrics.json",
        "classification_report.csv",
        "confusion_matrix.csv",
    ):
        target = destination / name
        try:
            os.link(source / name, target)
        except OSError:
            shutil.copy2(source / name, target)
    atomic_json(
        destination / "config.json",
        {"role": role, "alias_of": str(source), "genome": genome.to_dict()},
    )


def _atomic_npz(path: Path, **arrays) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".part.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _save_torch(path: Path, value) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    temporary.replace(path)


def _reports(targets, predictions, class_names: list[str]) -> tuple[list[dict], list[dict]]:
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            report_rows.append({"label": label, **values})
        else:
            report_rows.append({"label": label, "accuracy": values})
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(class_names)))
    matrix_rows = [
        {"true_label": class_names[row], **{class_names[col]: int(matrix[row, col]) for col in range(len(class_names))}}
        for row in range(len(class_names))
    ]
    return report_rows, matrix_rows


def refit_dataset(dataset_dir: Path, profile: bool = True) -> list[dict]:
    import torch

    manifest_path = dataset_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest["dataset"] == "SyntheticUEA":
        raise ValueError("Synthetic smoke runs cannot be refit against UEA TEST")
    representatives = load_json(dataset_dir / "search" / "representatives.json")
    config = manifest["config"]
    train = load_train(config["data_dir"], manifest["dataset"])
    # This is deliberately the first and only TEST access in the experiment lifecycle.
    test = load_test_for_refit(config["data_dir"], manifest["dataset"], train.class_names)
    stats = fit_normalization(train.samples)
    _atomic_npz(dataset_dir / "final" / "normalization_stats.npz", mean=stats.mean, std=stats.std)
    train_samples = normalize_samples(train.samples, stats)
    test_samples = normalize_samples(test.samples, stats)
    summary = []
    completed_hash_seeds: dict[tuple[str, int], Path] = {}
    for role, selected in representatives.items():
        genome = Genome(**{name: selected[name] for name in Genome.__dataclass_fields__}).validate()
        for seed in FINAL_SEEDS:
            output = ensure_dir(dataset_dir / "final" / role / str(seed))
            alias_key = (genome.sha256, seed)
            if alias_key in completed_hash_seeds:
                _materialize_alias(completed_hash_seeds[alias_key], output, role, genome)
                source_metrics = load_json(completed_hash_seeds[alias_key] / "metrics.json")
                summary.append({"role": role, "seed": seed, **source_metrics, "aliased": True})
                continue
            seed_everything(seed)
            train_loader = make_loader(train_samples, train.labels, config["batch_size"], True, seed)
            test_loader = make_loader(test_samples, test.labels, config["batch_size"], False, seed)
            model = MambaTSCClassifier(
                train.input_channels,
                len(train.class_names),
                genome,
                dropout=config.get("dropout", 0.1),
            )
            result = train_full_classifier(
                model,
                train_loader,
                test_loader,
                config["device"],
                config["epochs"],
                config["learning_rate"],
                config["weight_decay"],
            )
            macs = estimate_macs(genome, max(train.max_length, test.max_length), train.input_channels, len(train.class_names))
            metrics = {
                **result.metrics,
                "training_seconds": result.training_seconds,
                "epochs_trained": len(result.history),
                "parameters": trainable_parameters(result.model),
                "macs": macs.total,
                "candidate_hash": genome.sha256,
            }
            if profile and config["device"].startswith("cuda") and torch.cuda.is_available():
                sample, mask, _ = next(iter(make_loader(test_samples[:1], test.labels[:1], 1, False, seed)))
                metrics.update(
                    profile_cuda(result.model, sample.to(config["device"]), mask.to(config["device"]))
                )
            report, matrix = _reports(result.targets, result.predictions, train.class_names)
            atomic_json(output / "config.json", {"role": role, "seed": seed, "genome": genome.to_dict()})
            atomic_json(output / "metrics.json", metrics)
            atomic_csv(output / "history.csv", result.history)
            atomic_csv(output / "classification_report.csv", report)
            atomic_csv(output / "confusion_matrix.csv", matrix)
            _atomic_npz(
                output / "test_predictions.npz",
                logits=result.logits,
                predictions=result.predictions,
                targets=result.targets,
            )
            _save_torch(output / "checkpoint.pt", result.model.state_dict())
            completed_hash_seeds[alias_key] = output
            summary.append({"role": role, "seed": seed, **metrics, "aliased": False})
    atomic_csv(dataset_dir / "final" / "summary.csv", summary)
    manifest["test_accessed"] = True
    manifest["refit_completed"] = True
    atomic_json(manifest_path, manifest)
    return summary


def resolve_run(output_dir: str | Path, run: str) -> Path:
    supplied = Path(run)
    if supplied.exists():
        return supplied.resolve()
    path = Path(output_dir) / run
    if not path.exists():
        raise FileNotFoundError(f"Could not find run {run!r} under {output_dir}")
    return path.resolve()


def refit_run(run_dir: Path, profile: bool = True) -> None:
    datasets = [path for path in run_dir.iterdir() if path.is_dir() and (path / "manifest.json").exists()]
    if not datasets:
        raise ValueError(f"No dataset manifests found in {run_dir}")
    for dataset_dir in datasets:
        refit_dataset(dataset_dir, profile=profile)
