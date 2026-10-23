"""Plot search-algorithm convergence and model efficiency comparisons.

The script builds an offline reference front from the union of completed
candidate evaluations. It never uses TEST labels for search metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .convergence_metrics import (
    front_metrics,
    hypervolume,
    hv_auc,
    nondominated_points,
    objective_array,
    target_crossing,
)

METHODS = ("random", "nsga2", "archive_local_search")
LABELS = {"random": "Random", "nsga2": "NSGA-II", "archive_local_search": "Archive + Local"}
COLORS = {"random": "#7f8c8d", "nsga2": "#4c78a8", "archive_local_search": "#d62728"}
DEFAULT_DATASETS = ("Heartbeat", "SpokenArabicDigits", "PEMS-SF")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str, default=np.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def _run_records(root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...], kfold_datasets: tuple[str, ...]) -> list[dict]:
    records = []
    for dataset in datasets:
        for dataset_dir in sorted(root.glob(f"*/{dataset}")):
            manifest_path = dataset_dir / "manifest.json"
            evaluations_path = dataset_dir / "search" / "evaluations.csv"
            if not manifest_path.exists() or not evaluations_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config", {})
            algorithm = config.get("search_algorithm", "nsga2")
            variant = config.get("search_variant", "baseline")
            seed = int(config.get("seed", 2021))
            if algorithm not in METHODS or seed not in seeds:
                continue
            expected_protocol = "kfold" if dataset in kfold_datasets else "holdout"
            if config.get("validation_strategy", "holdout") != expected_protocol:
                continue
            rows = [row for row in _read_csv(evaluations_path) if row.get("status", "completed") == "completed"]
            if not rows:
                continue
            records.append({
                "dataset": dataset, "run": dataset_dir, "algorithm": algorithm,
                "variant": variant, "seed": seed, "rows": rows,
            })
    return records


def _reference_by_dataset(records: list[dict]) -> dict[str, np.ndarray]:
    references = {}
    for dataset in sorted({item["dataset"] for item in records}):
        rows = [row for item in records if item["dataset"] == dataset for row in item["rows"]]
        references[dataset] = nondominated_points(objective_array(rows))
        for item in records:
            if item["dataset"] != dataset:
                continue
            path = item["run"] / "search" / "reference_front.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["objective_f1", "objective_macs"])
                writer.writeheader()
                writer.writerows({"objective_f1": point[0], "objective_macs": point[1]} for point in references[dataset])
    return references


def _generation_curves(item: dict, reference: np.ndarray, budget: int, hv_reference: np.ndarray | None = None) -> list[dict]:
    search = item["run"] / "search"
    generation_paths = sorted((search / "generations").glob("generation_*_pareto.csv"))
    curves = []
    if hv_reference is None:
        max_macs = max((float(row.get("objective_macs", 0.0)) for row in item["rows"]), default=1.0)
        hv_reference = np.asarray([1.1, max(1.0, max_macs * 1.01)])
    for path in generation_paths:
        try:
            generation = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        front_rows = _read_csv(path)
        points = objective_array(front_rows)
        if len(points) == 0:
            continue
        metrics = front_metrics(points, reference, hv_reference)
        snapshot_path = search / "generations" / f"generation_{generation:04d}.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else {}
        curves.append({
            "dataset": item["dataset"], "algorithm": item["algorithm"], "variant": item["variant"],
            "seed": item["seed"], "generation": generation,
            "unique_evaluations": _float(snapshot, "unique_evaluations", generation * budget),
            "elapsed_seconds": _float(snapshot, "elapsed_seconds"),
            "training_seconds": _float(snapshot, "training_seconds"),
            "best_macro_f1": max((_float(row, "macro_f1", 0.0) for row in front_rows), default=0.0),
            "low_macs": min((_float(row, "macs", 0.0) for row in front_rows), default=0.0),
            **metrics,
        })
    return curves


def _aggregate_curves(records: list[dict], references: dict[str, np.ndarray], budget: int) -> tuple[list[dict], list[dict]]:
    curves, summaries = [], []
    for item in records:
        global_max_macs = max(
            (float(candidate.get("objective_macs", 0.0))
             for other in records if other["dataset"] == item["dataset"]
             for candidate in other["rows"]),
            default=1.0,
        )
        common_hv_reference = np.asarray([1.1, max(1.0, global_max_macs * 1.01)])
        rows = _generation_curves(item, references[item["dataset"]], budget, common_hv_reference)
        curves.extend(rows)
        if not rows:
            continue
        rows.sort(key=lambda row: row["unique_evaluations"])
        x = [row["unique_evaluations"] for row in rows]
        hv = [row["hypervolume"] for row in rows]
        for index, row in enumerate(rows):
            row["hypervolume_auc"] = hv_auc(x[: index + 1], hv[: index + 1], budget)
        convergence_path = item["run"] / "search" / "convergence_metrics.csv"
        fields = [
            "search_algorithm", "search_variant", "generation", "unique_evaluations",
            "elapsed_seconds", "training_seconds", "hypervolume", "hypervolume_auc",
            "gd", "igd", "igd_plus", "epsilon_indicator", "spacing", "spread",
            "pareto_size", "best_macro_f1", "low_macs",
        ]
        with convergence_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for row in rows:
                writer.writerow({
                    "search_algorithm": row["algorithm"], "search_variant": row["variant"],
                    **{key: row.get(key, "") for key in fields if key not in {"search_algorithm", "search_variant"}},
                })
        final = rows[-1]
        final_ref_hv = max(hypervolume(references[item["dataset"]], common_hv_reference), 1e-12)
        summaries.append({
            "dataset": item["dataset"], "algorithm": item["algorithm"], "variant": item["variant"], "seed": item["seed"],
            "final_hypervolume": final["hypervolume"], "hypervolume_auc": hv_auc(x, hv, budget),
            "final_igd_plus": final["igd_plus"], "final_epsilon_indicator": final["epsilon_indicator"],
            "pareto_size": final["pareto_size"], "unique_evaluations": x[-1],
            "training_seconds": final["training_seconds"], "elapsed_seconds": final["elapsed_seconds"],
            "evaluations_to_90_hv": target_crossing(x, hv, 0.90 * final_ref_hv),
            "evaluations_to_95_hv": target_crossing(x, hv, 0.95 * final_ref_hv),
            "seconds_to_90_hv": target_crossing([row["elapsed_seconds"] for row in rows], hv, 0.90 * final_ref_hv),
            "seconds_to_95_hv": target_crossing([row["elapsed_seconds"] for row in rows], hv, 0.95 * final_ref_hv),
        })
    return curves, summaries


def _save(fig, path: Path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_curves(curves: list[dict], output: Path, field: str, ylabel: str, filename: str, lower_better: bool = False):
    import matplotlib.pyplot as plt
    datasets = sorted({row["dataset"] for row in curves})
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for algorithm in METHODS:
            for seed in sorted({row["seed"] for row in curves if row["dataset"] == dataset and row["algorithm"] == algorithm}):
                rows = [row for row in curves if row["dataset"] == dataset and row["algorithm"] == algorithm and row["seed"] == seed]
                rows.sort(key=lambda row: row["unique_evaluations"])
                ax.plot([row["unique_evaluations"] for row in rows], [row[field] for row in rows], color=COLORS[algorithm], alpha=0.25, linewidth=1)
            rows = [row for row in curves if row["dataset"] == dataset and row["algorithm"] == algorithm]
            if rows:
                grouped = {}
                for row in rows:
                    grouped.setdefault(row["unique_evaluations"], []).append(row[field])
                x = sorted(grouped)
                y = [float(np.nanmean(grouped[value])) for value in x]
                ax.plot(x, y, color=COLORS[algorithm], linewidth=2, label=LABELS[algorithm])
        ax.set_title(dataset)
        ax.set_xlabel("Unique candidate evaluations")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(ylabel + " (mean curve; translucent lines are seeds)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output / filename)


def _plot_time_curves(curves: list[dict], output: Path):
    import matplotlib.pyplot as plt
    datasets = sorted({row["dataset"] for row in curves})
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for algorithm in METHODS:
            rows = [row for row in curves if row["dataset"] == dataset and row["algorithm"] == algorithm]
            for seed in sorted({row["seed"] for row in rows}):
                seed_rows = sorted((row for row in rows if row["seed"] == seed), key=lambda row: row["elapsed_seconds"])
                ax.plot([row["elapsed_seconds"] for row in seed_rows], [row["hypervolume"] for row in seed_rows], color=COLORS[algorithm], alpha=0.25, linewidth=1)
            if rows:
                ax.plot([], [], color=COLORS[algorithm], linewidth=2, label=LABELS[algorithm])
        ax.set_title(dataset); ax.set_xlabel("Wall-clock seconds"); ax.set_ylabel("Hypervolume")
        ax.grid(alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Hypervolume vs wall-clock time", fontsize=14, fontweight="bold")
    fig.tight_layout(); _save(fig, output / "search_hypervolume_vs_time.png")


def _parse_result_file(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            values[key.strip()] = value.strip()
    return values


def _efficiency_rows(root: Path, records: list[dict], role: str) -> list[dict]:
    rows = []
    for item in records:
        path = item["run"] / "final" / role / "2021" / "metrics.json"
        if not path.exists():
            continue
        values = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"dataset": item["dataset"], "model": item["algorithm"], "role": role, **values})
    result_root = root.parent / "results" if root.name == "runs" else Path("results")
    for path in sorted(result_root.glob("classification_*/result_classification.txt")):
        values = _parse_result_file(path)
        if "macro_f1" not in values:
            continue
        name = path.parent.name.removeprefix("classification_")
        model = name.split("_", 2)[1] if "_" in name else name
        rows.append({"dataset": name.split("_", 1)[0], "model": model, "role": "baseline", **values})
    return rows


def _plot_efficiency(rows: list[dict], output: Path):
    import matplotlib.pyplot as plt
    valid = [row for row in rows if np.isfinite(float(row.get("macro_f1", np.nan))) and np.isfinite(float(row.get("flops", row.get("analytical_flops", 2 * float(row.get("macs", np.nan))))))]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in valid:
        x = float(row.get("flops", row.get("analytical_flops", 2 * float(row.get("macs")))))
        y = float(row["macro_f1"])
        label = f"{row['model']} / {row['dataset']}"
        ax.scatter(x, y, s=70, alpha=0.85, label=label)
    ax.set_xscale("log"); ax.set_xlabel("FLOPs"); ax.set_ylabel("TEST Macro-F1")
    ax.grid(alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); _save(fig, output / "efficiency_f1_flops.png")


def _plot_efficiency_macs(rows: list[dict], output: Path):
    import matplotlib.pyplot as plt
    valid = [row for row in rows if np.isfinite(float(row.get("macro_f1", np.nan))) and np.isfinite(float(row.get("macs", np.nan)))]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in valid:
        ax.scatter(float(row["macs"]), float(row["macro_f1"]), s=70, alpha=0.85,
                   label=f"{row['model']} / {row['dataset']}")
    ax.set_xscale("log"); ax.set_xlabel("MACs"); ax.set_ylabel("TEST Macro-F1")
    ax.grid(alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); _save(fig, output / "efficiency_f1_macs.png")


def _plot_latency_memory(rows: list[dict], output: Path):
    import matplotlib.pyplot as plt
    valid = [row for row in rows if np.isfinite(float(row.get("latency_median_ms", np.nan))) and np.isfinite(float(row.get("peak_memory_bytes", np.nan)))]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in valid:
        params = float(row.get("parameters", 1) or 1)
        ax.scatter(float(row["latency_median_ms"]), float(row["peak_memory_bytes"]) / 1e6,
                   s=max(30, min(500, params / 1000)), alpha=0.85,
                   label=f"{row['model']} / {row['dataset']}")
    ax.set_xlabel("Latency median (ms, batch=1)"); ax.set_ylabel("Peak memory (MB)")
    ax.grid(alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); _save(fig, output / "efficiency_latency_memory.png")


def _plot_search_summaries(summaries: list[dict], output: Path):
    import matplotlib.pyplot as plt
    if not summaries:
        return
    datasets = sorted({row["dataset"] for row in summaries})
    metrics = (("evaluations_to_90_hv", "Evaluations to 90% HV"), ("seconds_to_90_hv", "Seconds to 90% HV"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
    for ax, (field, title) in zip(axes[0], metrics):
        for algorithm in METHODS:
            values = []
            labels = []
            for dataset in datasets:
                selected = [float(row[field]) for row in summaries if row["dataset"] == dataset and row["algorithm"] == algorithm and row.get(field) not in (None, "") and np.isfinite(float(row[field]))]
                if selected:
                    values.append(float(np.mean(selected))); labels.append(dataset)
            if values:
                x = np.arange(len(values)) + (list(METHODS).index(algorithm) - 1) * 0.18
                ax.bar(x, values, width=0.18, color=COLORS[algorithm], label=LABELS[algorithm])
                ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Search convergence speed summary", fontsize=14, fontweight="bold")
    fig.tight_layout(); _save(fig, output / "search_convergence_speed.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    for algorithm in METHODS:
        selected = [row for row in summaries if row["algorithm"] == algorithm]
        if not selected:
            continue
        x = [row["unique_evaluations"] for row in selected]
        y = [row["training_seconds"] for row in selected]
        ax.scatter(x, y, color=COLORS[algorithm], label=LABELS[algorithm], alpha=0.8)
    ax.set_xlabel("Unique evaluations"); ax.set_ylabel("Cumulative training seconds")
    ax.grid(alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False); fig.tight_layout(); _save(fig, output / "search_cost_summary.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs")
    parser.add_argument("--output", default="results/search_algorithm_comparison")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--kfold-datasets", nargs="+", default=[])
    parser.add_argument("--role", default="knee", choices=("high_f1", "knee", "low_macs"))
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    records = _run_records(root, tuple(args.datasets), tuple(args.seeds), tuple(args.kfold_datasets))
    if not records:
        raise SystemExit(f"No completed algorithm comparison runs found under {root}")
    references = _reference_by_dataset(records)
    curves, summaries = _aggregate_curves(records, references, args.budget)
    _plot_curves(curves, output, "hypervolume", "Hypervolume", "search_hypervolume_vs_evaluations.png")
    _plot_curves(curves, output, "igd_plus", "IGD+", "search_igd_plus_vs_evaluations.png", lower_better=True)
    _plot_curves(curves, output, "epsilon_indicator", "Additive epsilon", "search_epsilon_vs_evaluations.png", lower_better=True)
    _plot_time_curves(curves, output)
    with (output / "search_algorithm_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(summaries[0]) if summaries else []
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    efficiency = _efficiency_rows(root, records, args.role)
    with (output / "efficiency_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in efficiency for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(efficiency)
    _plot_efficiency(efficiency, output)
    _plot_efficiency_macs(efficiency, output)
    _plot_latency_memory(efficiency, output)
    _plot_search_summaries(summaries, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
