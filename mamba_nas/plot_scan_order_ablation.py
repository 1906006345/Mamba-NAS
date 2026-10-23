"""Generate scan-order/coupled-local-search ablation tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    "baseline",
    "scan_order",
    "scan_order_local_search",
    "scan_order_coupled_local_search",
)
LABELS = {
    "baseline": "NSGA-II",
    "scan_order": "Scan order",
    "scan_order_local_search": "Scan + Local",
    "scan_order_coupled_local_search": "Scan + Coupled Local",
}
COLORS = {
    "baseline": "#7F8C8D",
    "scan_order": "#4C78A8",
    "scan_order_local_search": "#F28E2B",
    "scan_order_coupled_local_search": "#D62728",
}
DATASETS = ("Heartbeat", "SpokenArabicDigits", "PEMS-SF")
ROLES = ("high_f1", "knee", "low_macs")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str, default=np.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value in (None, "", "nan") else float(value)
    except (TypeError, ValueError):
        return default


def load_records(root: Path, seeds: tuple[int, ...]) -> dict:
    records = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in seeds:
                pattern = f"{dataset.lower()}-holdout-scan-ablation-{variant}-seed{seed}/{dataset}"
                matches = sorted(root.glob(pattern))
                if not matches:
                    continue
                run = matches[-1]
                search = run / "search"
                generations = read_csv(search / "generation_diversity.csv")
                generations.sort(key=lambda row: int(float(row["generation"])))
                front = read_csv(search / "pareto_front.csv")
                local = read_csv(search / "local_search.csv") if (search / "local_search.csv").exists() else []
                evaluations = [
                    row for row in read_csv(search / "evaluations.csv")
                    if row.get("status", "completed") == "completed"
                ]
                final_path = run / "final" / "summary.csv"
                final = read_csv(final_path) if final_path.exists() else []
                records[(dataset, variant, seed)] = {
                    "run": run, "generations": generations, "front": front,
                    "local": local, "final": final, "evaluations": evaluations,
                }
    if not records:
        raise FileNotFoundError(f"No scan-order ablation runs found under {root}")
    return records


def save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax) -> None:
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_convergence(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4.8), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            items = [item for (d, v, _), item in records.items() if d == dataset and v == variant]
            for item in items:
                rows = item["generations"]
                ax.plot(
                    [number(row, "unique_evaluations") for row in rows],
                    [number(row, "hypervolume") for row in rows],
                    color=COLORS[variant], alpha=0.25, linewidth=1,
                )
            if items:
                xs = sorted({number(row, "unique_evaluations") for item in items for row in item["generations"]})
                ys = []
                for x in xs:
                    values = [number(row, "hypervolume") for item in items for row in item["generations"] if number(row, "unique_evaluations") == x]
                    ys.append(float(np.mean(values)))
                ax.plot(xs, ys, color=COLORS[variant], linewidth=2.2, label=LABELS[variant])
        ax.set_title(dataset); ax.set_xlabel("Unique evaluations"); ax.set_ylabel("Hypervolume"); style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Scan-order ablation: Hypervolume convergence", fontweight="bold")
    fig.tight_layout(); save(fig, output / "scan_order_hv_convergence.png")


def final_rows(records: dict) -> list[dict]:
    rows = []
    for (dataset, variant, seed), item in records.items():
        last = item["generations"][-1]
        counts = {order: sum(row.get("scan_order", "time_major") == order for row in item["front"])
                  for order in ("time_major", "patch_interleaved")}
        rows.append({
            "dataset": dataset, "variant": variant, "seed": seed,
            "final_hypervolume": number(last, "hypervolume"),
            "hypervolume_auc": number(last, "hypervolume_auc"),
            "pareto_size": len(item["front"]),
            "structure_coverage": number(last, "structure_coverage"),
            "scan_order_coverage": number(last, "scan_order_coverage"),
            "scan_order_entropy": number(last, "scan_order_entropy"),
            "mean_genome_distance": number(last, "mean_genome_distance"),
            "time_major_pareto_count": counts["time_major"],
            "patch_interleaved_pareto_count": counts["patch_interleaved"],
            "single_pareto_entry_rate": number(last, "single_pareto_entry_rate", 0.0),
            "coupled_pareto_entry_rate": number(last, "coupled_pareto_entry_rate", 0.0),
            "single_mean_delta_f1": number(last, "single_mean_delta_f1", 0.0),
            "coupled_mean_delta_f1": number(last, "coupled_mean_delta_f1", 0.0),
        })
    return rows


def plot_bars(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    metrics = (("hypervolume_auc", "HV-AUC", "scan_order_hv_auc.png"),
               ("scan_order_coverage", "Scan-order coverage", "scan_order_coverage.png"))
    for metric, title, filename in metrics:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.5), squeeze=False)
        for ax, dataset in zip(axes[0], DATASETS):
            values, errors = [], []
            for variant in VARIANTS:
                selected = [row[metric] for row in rows if row["dataset"] == dataset and row["variant"] == variant]
                values.append(np.nanmean(selected) if selected else np.nan)
                errors.append(np.nanstd(selected, ddof=1) if len(selected) > 1 else 0.0)
            x = np.arange(len(VARIANTS))
            ax.bar(x, values, yerr=errors, color=[COLORS[v] for v in VARIANTS], capsize=3)
            ax.set_xticks(x, [LABELS[v] for v in VARIANTS], rotation=25, ha="right", fontsize=8)
            ax.set_title(dataset); ax.set_ylabel(title); style(ax)
        fig.suptitle(title, fontweight="bold"); fig.tight_layout(); save(fig, output / filename)


def plot_frequency(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        time_values, interleaved_values = [], []
        for variant in VARIANTS:
            selected = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            time_values.append(sum(row["time_major_pareto_count"] for row in selected))
            interleaved_values.append(sum(row["patch_interleaved_pareto_count"] for row in selected))
        x = np.arange(len(VARIANTS))
        ax.bar(x, time_values, color="#4C78A8", label="time_major")
        ax.bar(x, interleaved_values, bottom=time_values, color="#F28E2B", label="patch_interleaved")
        ax.set_xticks(x, [LABELS[v] for v in VARIANTS], rotation=25, ha="right", fontsize=8)
        ax.set_title(dataset); ax.set_ylabel("Pareto occurrences across seeds"); style(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Scan-order frequency on final Pareto fronts", fontweight="bold")
    fig.tight_layout(); save(fig, output / "scan_order_frequency.png")


def plot_pareto(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4.8), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            items = [item for (d, v, _), item in records.items() if d == dataset and v == variant]
            for item in items:
                ax.scatter(
                    [number(row, "macs") for row in item["evaluations"]],
                    [number(row, "macro_f1") for row in item["evaluations"]],
                    s=5, color=COLORS[variant], alpha=0.025,
                )
                front = sorted(item["front"], key=lambda row: number(row, "macs"))
                ax.plot(
                    [number(row, "macs") for row in front],
                    [number(row, "macro_f1") for row in front],
                    color=COLORS[variant], linewidth=1, alpha=0.35,
                )
            if items:
                ax.plot([], [], color=COLORS[variant], linewidth=2, label=LABELS[variant])
        ax.set_xscale("log"); ax.set_title(dataset); ax.set_xlabel("MACs"); ax.set_ylabel("Validation Macro-F1"); style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Scan-order ablation Pareto fronts", fontweight="bold")
    fig.tight_layout(); save(fig, output / "scan_order_pareto_front.png")


def plot_local_diagnostics(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    variant = "scan_order_coupled_local_search"
    metrics = (("single_pareto_entry_rate", "Single entry rate"),
               ("coupled_pareto_entry_rate", "Coupled entry rate"),
               ("single_mean_delta_f1", "Single mean ΔF1"),
               ("coupled_mean_delta_f1", "Coupled mean ΔF1"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for ax, (metric, title) in zip(axes.flat, metrics):
        selected = [[row[metric] for row in rows if row["dataset"] == dataset and row["variant"] == variant] for dataset in DATASETS]
        means = [np.mean(values) if values else np.nan for values in selected]
        stds = [np.std(values, ddof=1) if len(values) > 1 else 0.0 for values in selected]
        ax.bar(np.arange(len(DATASETS)), means, yerr=stds, color="#D62728", capsize=3)
        ax.set_xticks(np.arange(len(DATASETS)), DATASETS, rotation=20, ha="right")
        ax.set_title(title); style(ax)
    fig.suptitle("Coupled Local Search diagnostics", fontweight="bold")
    fig.tight_layout(); save(fig, output / "coupled_local_search_diagnostics.png")


def plot_test(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4.8), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        x = np.arange(len(VARIANTS)); width = 0.24
        for role_index, role in enumerate(ROLES):
            means, stds = [], []
            for variant in VARIANTS:
                values = [number(row, "macro_f1") for (d, v, _), item in records.items()
                          if d == dataset and v == variant for row in item["final"] if row.get("role") == role]
                means.append(np.mean(values) if values else np.nan)
                stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            ax.bar(x + (role_index - 1) * width, means, width, yerr=stds, capsize=2,
                   label=role, alpha=0.85)
        ax.set_xticks(x, [LABELS[v] for v in VARIANTS], rotation=25, ha="right", fontsize=8)
        ax.set_title(dataset); ax.set_ylabel("TEST Macro-F1"); style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Refit TEST Macro-F1", fontweight="bold")
    fig.tight_layout(); save(fig, output / "scan_order_test_f1.png")

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 4.8), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            for role in ROLES:
                selected = [row for (d, v, _), item in records.items() if d == dataset and v == variant
                            for row in item["final"] if row.get("role") == role]
                if selected:
                    ax.scatter(np.mean([number(row, "macs") for row in selected]),
                               np.mean([number(row, "macro_f1") for row in selected]),
                               color=COLORS[variant], alpha=0.8, label=LABELS[variant])
        ax.set_xscale("log"); ax.set_title(dataset); ax.set_xlabel("MACs"); ax.set_ylabel("TEST Macro-F1"); style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles)); fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, output / "scan_order_test_f1_macs.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs")
    parser.add_argument("--output", default="results/scan_order_ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    records = load_records(root, tuple(args.seeds)); rows = final_rows(records)
    plot_convergence(records, output); plot_bars(rows, output); plot_pareto(records, output); plot_frequency(rows, output)
    plot_local_diagnostics(rows, output); plot_test(records, output)
    with (output / "scan_order_ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
