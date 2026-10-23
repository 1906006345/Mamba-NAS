"""Plot the focused baseline/dual-space weekly-report figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .plot_ablation_search import (
    COLORS,
    DATASETS,
    LABELS,
    ROLE_LABELS,
    ROLES,
    _float,
    _load_records,
    _role_stats,
    _save,
    _style,
)


VARIANTS = ("baseline", "dual_crowding", "diversity_archive", "dual_space")


def plot_search_convergence(records: dict, output: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            rows = records[(dataset, variant)]["generations"]
            ax.plot(
                [_float(row, "unique_evaluations") for row in rows],
                [_float(row, "hypervolume") for row in rows],
                color=COLORS[variant],
                linewidth=2.2,
                marker="o",
                markersize=2.8,
                label=LABELS[variant],
            )
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("Unique candidate evaluations")
        ax.set_ylabel("Hypervolume")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Search convergence (NAS seed {seed})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    _save(fig, output / "weekly_search_convergence.png")


def plot_search_metrics(records: dict, output: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("final_hv", "Final Hypervolume"),
        ("final_structure_coverage", "Structure coverage (18 niches)"),
        ("final_direction_coverage", "Direction coverage"),
        ("final_mean_genome_distance", "Mean weighted Genome distance"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), squeeze=False)
    positions = np.arange(len(VARIANTS))
    width = 0.37
    for ax, (metric, title) in zip(axes.flat, metrics):
        for dataset_index, dataset in enumerate(DATASETS):
            values = [records[(dataset, variant)][metric] for variant in VARIANTS]
            offset = (-0.5 if dataset_index == 0 else 0.5) * width
            bars = ax.bar(
                positions + offset,
                values,
                width,
                color="#4C78A8" if dataset_index == 0 else "#F28E2B",
                alpha=0.86,
                label=dataset,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        ax.set_title(title)
        ax.set_xticks(positions, [LABELS[v] for v in VARIANTS], rotation=24, ha="right")
        _style(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Search quality and structural diversity (NAS seed {seed})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "weekly_search_metrics.png")


def plot_test_performance(records: dict, output: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), squeeze=False)
    positions = np.arange(len(VARIANTS))
    width = 0.24
    role_colors = {"high_f1": "#4C78A8", "knee": "#59A14F", "low_macs": "#F28E2B"}
    for ax, dataset in zip(axes[0], DATASETS):
        all_means = []
        for role_index, role in enumerate(ROLES):
            means, stds = [], []
            for variant in VARIANTS:
                stats = _role_stats(records[(dataset, variant)], role)
                means.append(stats["macro_f1_mean"])
                stds.append(stats.get("macro_f1_std", 0.0))
            all_means.extend(means)
            ax.bar(
                positions + (role_index - 1) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=role_colors[role],
                alpha=0.88,
                label=ROLE_LABELS[role],
            )
        margin = max(0.008, (max(all_means) - min(all_means)) * 0.3)
        ax.set_ylim(max(0, min(all_means) - margin), min(1.005, max(all_means) + margin))
        ax.set_title(dataset, fontweight="bold")
        ax.set_ylabel("TEST Macro-F1 (mean ± std)")
        ax.set_xticks(positions, [LABELS[v] for v in VARIANTS], rotation=24, ha="right")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Refit TEST performance (NAS seed {seed}; refit seeds 2021/2022/2023)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    _save(fig, output / "weekly_test_performance.png")


def plot_test_tradeoff(records: dict, output: Path, seed: int) -> None:
    import matplotlib.pyplot as plt

    markers = {"high_f1": "*", "knee": "D", "low_macs": "P"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            for role in ROLES:
                stats = _role_stats(records[(dataset, variant)], role)
                ax.errorbar(
                    stats["macs_mean"],
                    stats["macro_f1_mean"],
                    yerr=stats.get("macro_f1_std", 0.0),
                    color=COLORS[variant],
                    marker=markers[role],
                    markersize=9,
                    linestyle="none",
                    capsize=2.5,
                    label=f"{LABELS[variant]} / {ROLE_LABELS[role]}",
                )
        ax.set_xscale("log")
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("MACs (log scale)")
        ax.set_ylabel("TEST Macro-F1 (mean ± std)")
        _style(ax)
    handles, labels = [], []
    for ax in axes[0]:
        h, label = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(label)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("TEST performance–cost trade-off", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    _save(fig, output / "weekly_test_f1_macs.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs")
    parser.add_argument("--output", default="results/weekly_dual_space_2021")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    records, _ = _load_records(Path(args.input).resolve(), args.seed)
    required = [(dataset, variant) for dataset in DATASETS for variant in VARIANTS]
    missing = [key for key in required if key not in records or not records[key].get("final_rows")]
    if missing:
        raise FileNotFoundError("Missing search/refit artifacts: " + ", ".join(f"{d}/{v}" for d, v in missing))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_search_convergence(records, output, args.seed)
    plot_search_metrics(records, output, args.seed)
    plot_test_performance(records, output, args.seed)
    plot_test_tradeoff(records, output, args.seed)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
