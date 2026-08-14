from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import atomic_csv, atomic_json, ensure_dir, load_json


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export_paper(run_dir: Path, export_root: str | Path = "paper_exports") -> Path:
    destination = ensure_dir(Path(export_root) / run_dir.name)
    all_candidates, pareto_rows, generation_rows, final_rows, confusion_rows = [], [], [], [], []
    manifests = []
    for dataset_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        dataset = manifest["dataset"]
        manifests.append(manifest)
        candidates = _read_csv(dataset_dir / "search" / "evaluations.csv")
        all_candidates.extend({"dataset": dataset, **row} for row in candidates)
        front = _read_csv(dataset_dir / "search" / "pareto_front.csv")
        pareto_rows.extend({"dataset": dataset, **row} for row in front)
        for generation_file in sorted((dataset_dir / "search" / "generations").glob("generation_*.json")):
            generation_rows.append({"dataset": dataset, **load_json(generation_file)})
        final_rows.extend(
            {"dataset": dataset, **row} for row in _read_csv(dataset_dir / "final" / "summary.csv")
        )
        for matrix_path in sorted((dataset_dir / "final").glob("*/*/confusion_matrix.csv")):
            role, seed = matrix_path.parts[-3], matrix_path.parts[-2]
            confusion_rows.extend(
                {"dataset": dataset, "role": role, "seed": seed, **row}
                for row in _read_csv(matrix_path)
            )
    atomic_csv(destination / "all_candidates.csv", all_candidates)
    atomic_csv(destination / "pareto_plot_data.csv", pareto_rows)
    atomic_csv(destination / "generation_hypervolume.csv", generation_rows)
    atomic_csv(destination / "final_seed_results.csv", final_rows)
    atomic_csv(destination / "confusion_matrices.csv", confusion_rows)

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in final_rows:
        for metric in ("macro_f1", "accuracy", "parameters", "macs", "training_seconds"):
            if row.get(metric) not in (None, ""):
                grouped[(row["dataset"], row["role"])][metric].append(float(row[metric]))
    aggregate = []
    for (dataset, role), metrics in sorted(grouped.items()):
        row = {"dataset": dataset, "role": role}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(row)
    atomic_csv(destination / "final_mean_std.csv", aggregate)

    ablations = []
    dimensions = ("tokenizer", "num_blocks", "direction", "d_model", "d_state", "d_conv", "expand", "pooling")
    for dataset in sorted({row["dataset"] for row in all_candidates}):
        subset = [row for row in all_candidates if row["dataset"] == dataset and row.get("status") == "completed"]
        for dimension in dimensions:
            for value in sorted({row[dimension] for row in subset}):
                selected = [row for row in subset if row[dimension] == value]
                ablations.append(
                    {
                        "dataset": dataset,
                        "search_object": dimension,
                        "value": value,
                        "n": len(selected),
                        "macro_f1_mean": float(np.mean([float(row["macro_f1"]) for row in selected])),
                        "macs_mean": float(np.mean([float(row["macs"]) for row in selected])),
                    }
                )
    atomic_csv(destination / "search_space_ablation.csv", ablations)
    atomic_json(destination / "manifests.json", manifests)
    atomic_json(
        destination / "README.json",
        {
            "regenerable_from": str(run_dir),
            "contains_weights": False,
            "contains_predictions": False,
            "files": [path.name for path in sorted(destination.iterdir()) if path.is_file()],
        },
    )
    return destination
