from __future__ import annotations

import csv
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from .artifacts import atomic_csv, atomic_json, ensure_dir, load_json
from .data import fit_normalization, load_test_for_refit, load_train, make_loader, normalize_samples
from .logging_utils import DEFAULT_RESULT_PATH, configure_logging, log_test_result
from .macs import estimate_macs, trainable_parameters
from .model import MambaTSCClassifier
from .profiling import profile_cuda
from .search_space import Genome, genome_from_mapping
from .training import seed_everything, train_full_classifier


LOGGER = logging.getLogger(__name__)


FINAL_SEEDS = (2021, 2022, 2023)
LEGACY_ROLE_NAMES = {"high_accuracy": "high_f1", "low_cost": "low_macs"}


def canonical_representatives(representatives: dict[str, dict]) -> dict[str, dict]:
    """Read old run artifacts while writing all new outputs with corrected role names."""
    canonical = {}
    for role, selected in representatives.items():
        name = LEGACY_ROLE_NAMES.get(role, role)
        if name in canonical and canonical[name]["candidate_hash"] != selected["candidate_hash"]:
            raise ValueError(f"Conflicting representative definitions for role {name!r}")
        canonical[name] = selected
    return canonical


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
    configure_logging(dataset_dir / "final" / "refit.log")
    if manifest["dataset"] == "SyntheticUEA":
        raise ValueError("Synthetic smoke runs cannot be refit against UEA TEST")
    LOGGER.info(
        "[重训练][数据集开始] dataset=%s path=%s profile=%s",
        manifest["dataset"],
        dataset_dir,
        profile,
    )
    representatives = canonical_representatives(
        load_json(dataset_dir / "search" / "representatives.json")
    )
    config = manifest["config"]
    train = load_train(config["data_dir"], manifest["dataset"])
    # This is deliberately the first and only TEST access in the experiment lifecycle.
    test = load_test_for_refit(config["data_dir"], manifest["dataset"], train.class_names)
    LOGGER.info(
        "[重训练][数据加载] dataset=%s train_samples=%d test_samples=%d classes=%d "
        "channels=%d epochs=%d device=%s",
        manifest["dataset"],
        len(train.samples),
        len(test.samples),
        len(train.class_names),
        train.input_channels,
        config["epochs"],
        config["device"],
    )
    stats = fit_normalization(train.samples)
    _atomic_npz(dataset_dir / "final" / "normalization_stats.npz", mean=stats.mean, std=stats.std)
    train_samples = normalize_samples(train.samples, stats)
    test_samples = normalize_samples(test.samples, stats)
    summary = []
    completed_hash_seeds: dict[tuple[str, int], Path] = {}
    for role, selected in representatives.items():
        genome = genome_from_mapping(selected)
        LOGGER.info(
            "[重训练][代表架构] dataset=%s role=%s candidate=%s genome=%s",
            manifest["dataset"],
            role,
            genome.sha256[:12],
            genome.to_dict(),
        )
        for seed in FINAL_SEEDS:
            output = ensure_dir(dataset_dir / "final" / role / str(seed))
            alias_key = (genome.sha256, seed)
            if alias_key in completed_hash_seeds:
                _materialize_alias(completed_hash_seeds[alias_key], output, role, genome)
                source_metrics = load_json(completed_hash_seeds[alias_key] / "metrics.json")
                summary.append({"role": role, "seed": seed, **source_metrics, "aliased": True})
                log_test_result(manifest["dataset"], role, seed, source_metrics, aliased=True, result_path=DEFAULT_RESULT_PATH)
                continue
            LOGGER.info(
                "[重训练][种子开始] dataset=%s role=%s seed=%d candidate=%s",
                manifest["dataset"],
                role,
                seed,
                genome.sha256[:12],
            )
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
                log_context=f"dataset={manifest['dataset']} role={role} seed={seed}",
            )
            macs = estimate_macs(genome, max(train.max_length, test.max_length), train.input_channels, len(train.class_names))
            metrics = {
                **result.metrics,
                "training_seconds": result.training_seconds,
                "epochs_trained": len(result.history),
                "parameters": trainable_parameters(result.model),
                "macs": macs.total,
                "flops": macs.flops,
                "analytical_macs": macs.total,
                "analytical_flops": macs.flops,
                "macs_source": "analytical_mamba_cost",
                "flops_source": "analytical_flops_2x_macs",
                "profile_macs": None,
                "profile_flops": None,
                "profile_flops_status": "not_profiled",
                "candidate_hash": genome.sha256,
            }
            if profile and config["device"].startswith("cuda") and torch.cuda.is_available():
                sample, mask, _ = next(iter(make_loader(test_samples[:1], test.labels[:1], 1, False, seed)))
                metrics.update(
                    profile_cuda(result.model, sample.to(config["device"]), mask.to(config["device"]))
                )
            metrics["efficiency_definition"] = (
                "analytical FLOPs=2x MACs; profile FLOPs from torch.profiler when available; "
                "latency batch=1"
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
            log_test_result(manifest["dataset"], role, seed, metrics, aliased=False, result_path=DEFAULT_RESULT_PATH)
    atomic_csv(dataset_dir / "final" / "summary.csv", summary)
    efficiency_fields = [
        "dataset", "role", "seed", "macro_f1", "accuracy", "parameters",
        "analytical_macs", "analytical_flops", "profile_macs", "profile_flops",
        "latency_median_ms", "latency_p90_ms", "throughput_samples_per_sec",
        "peak_memory_bytes", "profile_flops_status", "efficiency_definition",
    ]
    atomic_csv(
        dataset_dir / "final" / "efficiency_summary.csv",
        [{"dataset": manifest["dataset"], **{key: row.get(key, "") for key in efficiency_fields if key != "dataset"}}
         for row in summary],
        fieldnames=efficiency_fields,
    )
    for role in representatives:
        role_rows = [row for row in summary if row["role"] == role]
        macro_f1 = np.asarray([row["macro_f1"] for row in role_rows], dtype=float)
        accuracy = np.asarray([row["accuracy"] for row in role_rows], dtype=float)
        LOGGER.info(
            "[TEST结果汇总] dataset=%s role=%s seeds=%d "
            "macro_f1_mean=%.6f macro_f1_std=%.6f "
            "accuracy_mean=%.6f accuracy_std=%.6f",
            manifest["dataset"],
            role,
            len(role_rows),
            float(macro_f1.mean()),
            float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0,
            float(accuracy.mean()),
            float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
        )
    manifest["test_accessed"] = True
    manifest["refit_completed"] = True
    atomic_json(manifest_path, manifest)
    LOGGER.info(
        "[重训练][数据集完成] dataset=%s runs=%d summary=%s",
        manifest["dataset"],
        len(summary),
        dataset_dir / "final" / "summary.csv",
    )
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
    LOGGER.info(
        "[重训练][开始] run=%s datasets=%d profile=%s",
        run_dir,
        len(datasets),
        profile,
    )
    for dataset_dir in sorted(datasets):
        refit_dataset(dataset_dir, profile=profile)
    LOGGER.info("[重训练][完成] run=%s datasets=%d", run_dir, len(datasets))
