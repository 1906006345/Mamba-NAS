"""Generate search and refit ablation figures for Mamba-NAS.

Search figures read ``search/`` artifacts.  When ``final/summary.csv`` exists,
the script additionally aggregates independent refit seeds and plots TEST
performance, efficiency, and the search-validation-to-TEST generalization gap.

Example:
    python -m mamba_nas.plot_ablation_search --output results/ablation_2021
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    "baseline",
    "dual_crowding",
    "diversity_archive",
    "local_search",
    "dual_space",
    "full",
)
DATASETS = ("Heartbeat", "SpokenArabicDigits")
COLORS = {
    "baseline": "#7F8C8D",
    "dual_crowding": "#4C78A8",
    "diversity_archive": "#59A14F",
    "local_search": "#F28E2B",
    "dual_space": "#B279A2",
    "full": "#D1495B",
}
LABELS = {
    "baseline": "Baseline",
    "dual_crowding": "Decision crowding",
    "diversity_archive": "Diversity archive",
    "local_search": "Local search",
    "dual_space": "Dual space",
    "full": "Full",
}
ROLES = ("high_f1", "knee", "low_macs")
ROLE_LABELS = {"high_f1": "High-F1", "knee": "Knee", "low_macs": "Low-MACs"}
ROLE_MARKERS = {"high_f1": "*", "knee": "D", "low_macs": "P"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def _find_run(input_root: Path, dataset: str, variant: str, seed: int) -> Path | None:
    patterns = (
        f"*-holdout-{variant}-seed{seed}/{dataset}",
        f"*-{variant}-seed{seed}/{dataset}",
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(input_root.glob(pattern)))
    for path in matches:
        if (path / "search" / "generation_diversity.csv").exists():
            return path
    return None


def _load_records(input_root: Path, seed: int) -> tuple[dict, list[dict]]:
    records = {}
    missing = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            run = _find_run(input_root, dataset, variant, seed)
            if run is None:
                missing.append((dataset, variant))
                continue
            search = run / "search"
            generations = _read_csv(search / "generation_diversity.csv")
            generations.sort(key=lambda row: int(float(row["generation"])))
            pareto = _read_csv(search / "pareto_front.csv")
            evaluations = [row for row in _read_csv(search / "evaluations.csv") if row.get("status", "completed") == "completed"]
            final = generations[-1] if generations else {}
            summary_path = run / "final" / "summary.csv"
            final_rows = _read_csv(summary_path) if summary_path.exists() else []
            representatives_path = search / "representatives.json"
            representatives = (
                json.loads(representatives_path.read_text(encoding="utf-8"))
                if representatives_path.exists()
                else {}
            )
            records[(dataset, variant)] = {
                "dataset": dataset,
                "variant": variant,
                "run": run,
                "generations": generations,
                "pareto": pareto,
                "evaluations": evaluations,
                "final_hv": _float(final, "hypervolume"),
                "final_structure_coverage": _float(final, "structure_coverage"),
                "final_tokenizer_coverage": _float(final, "tokenizer_coverage"),
                "final_direction_coverage": _float(final, "direction_coverage"),
                "final_mean_genome_distance": _float(final, "mean_genome_distance"),
                "unique_evaluations": _float(final, "unique_evaluations"),
                "pareto_size": len(pareto),
                "final_rows": final_rows,
                "representatives": representatives,
            }
    if missing:
        print("Warning: missing search runs:", ", ".join(f"{d}/{v}" for d, v in missing))
    if not records:
        raise FileNotFoundError(f"No seed-{seed} ablation artifacts found under {input_root}")
    return records, missing


def _style(ax):
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path: Path):
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_convergence(records: dict, output: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 5), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            item = records.get((dataset, variant))
            if not item:
                continue
            rows = item["generations"]
            ax.plot(
                [_float(row, "unique_evaluations") for row in rows],
                [_float(row, "hypervolume") for row in rows],
                marker="o",
                markersize=2.5,
                linewidth=1.8,
                color=COLORS[variant],
                label=LABELS[variant],
            )
        ax.set_title(dataset)
        ax.set_xlabel("Unique candidate evaluations")
        ax.set_ylabel("Hypervolume")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Search convergence (NAS seed 2021)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    _save(fig, output / "search_convergence_hypervolume_2021.png")


def _plot_diversity(records: dict, output: Path):
    import matplotlib.pyplot as plt

    metrics = (
        ("final_hv", "Final Hypervolume", False),
        ("final_structure_coverage", "Structure coverage", False),
        ("final_tokenizer_coverage", "Tokenizer coverage", False),
        ("final_direction_coverage", "Direction coverage", False),
        ("final_mean_genome_distance", "Mean Genome distance", False),
        ("pareto_size", "Pareto size", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    positions = np.arange(len(VARIANTS))
    for ax, (metric, title, _) in zip(axes.flat, metrics):
        width = 0.38
        for offset, dataset in zip((-width / 2, width / 2), DATASETS):
            values = [records.get((dataset, variant), {}).get(metric, np.nan) for variant in VARIANTS]
            ax.bar(positions + offset, values, width, label=dataset, color="#4C78A8" if offset < 0 else "#F28E2B", alpha=0.86)
        ax.set_title(title)
        ax.set_xticks(positions, [LABELS[v] for v in VARIANTS], rotation=35, ha="right", fontsize=8)
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Search quality and architecture diversity (NAS seed 2021)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "search_quality_diversity_2021.png")


def _plot_pareto(records: dict, output: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 6), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            item = records.get((dataset, variant))
            if not item:
                continue
            candidates = item["evaluations"]
            ax.scatter(
                [_float(row, "macs") for row in candidates],
                [_float(row, "macro_f1") for row in candidates],
                s=8,
                alpha=0.07,
                color=COLORS[variant],
            )
            front = item["pareto"]
            front = sorted(front, key=lambda row: _float(row, "macs"))
            if front:
                ax.plot(
                    [_float(row, "macs") for row in front],
                    [_float(row, "macro_f1") for row in front],
                    marker="o",
                    markersize=3.5,
                    linewidth=2,
                    color=COLORS[variant],
                    label=f"{LABELS[variant]} (n={len(front)})",
                )
        ax.set_xscale("log")
        ax.set_title(dataset)
        ax.set_xlabel("MACs (log scale)")
        ax.set_ylabel("Validation Macro-F1")
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Pareto-front comparison (search validation only, seed 2021)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "pareto_front_ablation_2021.png")


def _plot_module_summary(records: dict, output: Path):
    import matplotlib.pyplot as plt

    rows = []
    for dataset in DATASETS:
        base = records.get((dataset, "baseline"))
        if not base:
            continue
        for variant in VARIANTS:
            item = records.get((dataset, variant))
            if not item:
                continue
            rows.append(
                [
                    dataset,
                    LABELS[variant],
                    f"{item['final_hv']:.4f}",
                    f"{item['final_hv'] - base['final_hv']:+.4f}",
                    f"{item['final_structure_coverage']:.3f}",
                    f"{item['final_direction_coverage']:.3f}",
                    f"{item['final_mean_genome_distance']:.2f}",
                    str(int(item["pareto_size"])),
                ]
            )
    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.42 * len(rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Dataset", "Variant", "Final HV", "ΔHV vs baseline", "Structure cov.", "Direction cov.", "Genome dist.", "Pareto n"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#34495E")
        elif col == 1:
            cell.set_facecolor(COLORS.get(next((v for v in VARIANTS if LABELS[v] == cell.get_text().get_text()), "baseline"), "#FFFFFF"))
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("Ablation summary: search-only evidence (NAS seed 2021)", pad=18, fontsize=16, fontweight="bold")
    _save(fig, output / "ablation_summary_table_2021.png")


def _role_stats(item: dict, role: str) -> dict | None:
    rows = [row for row in item.get("final_rows", []) if row.get("role") == role]
    if not rows:
        return None
    result = {"n": len(rows)}
    for metric in (
        "macro_f1", "accuracy", "parameters", "macs", "latency_median_ms",
        "peak_memory_bytes", "training_seconds",
    ):
        values = np.asarray([_float(row, metric) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    representative = item.get("representatives", {}).get(role, {})
    result["search_macro_f1"] = _float(representative, "macro_f1")
    result["candidate_hash"] = rows[0].get("candidate_hash", "")
    return result


def _plot_test_performance(records: dict, output: Path, seed: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(17, 9), squeeze=False)
    positions = np.arange(len(VARIANTS), dtype=float)
    width = 0.24
    role_colors = {"high_f1": "#4C78A8", "knee": "#59A14F", "low_macs": "#F28E2B"}
    for row_index, dataset in enumerate(DATASETS):
        for column_index, metric in enumerate(("macro_f1", "accuracy")):
            ax = axes[row_index, column_index]
            for role_index, role in enumerate(ROLES):
                means, stds = [], []
                for variant in VARIANTS:
                    item = records.get((dataset, variant))
                    stats = _role_stats(item, role) if item else None
                    means.append(stats.get(f"{metric}_mean", np.nan) if stats else np.nan)
                    stds.append(stats.get(f"{metric}_std", 0.0) if stats else 0.0)
                offset = (role_index - 1) * width
                ax.bar(
                    positions + offset, means, width, yerr=stds, capsize=2.5,
                    color=role_colors[role], alpha=0.88, label=ROLE_LABELS[role],
                )
            finite = [
                _role_stats(records[(dataset, variant)], role).get(f"{metric}_mean", np.nan)
                for variant in VARIANTS if (dataset, variant) in records
                for role in ROLES if _role_stats(records[(dataset, variant)], role)
            ]
            finite = [value for value in finite if np.isfinite(value)]
            if finite:
                margin = max(0.01, (max(finite) - min(finite)) * 0.35)
                ax.set_ylim(max(0, min(finite) - margin), min(1.01, max(finite) + margin))
            ax.set_title(f"{dataset} — TEST {metric.replace('_', ' ').title()}")
            ax.set_ylabel("Mean ± std over refit seeds")
            ax.set_xticks(positions, [LABELS[v] for v in VARIANTS], rotation=30, ha="right", fontsize=8)
            _style(ax)
            missing = [LABELS[v] for v in VARIANTS if not records.get((dataset, v), {}).get("final_rows")]
            if missing:
                ax.text(0.01, 0.02, "Missing: " + ", ".join(missing), transform=ax.transAxes, fontsize=8, color="#B03A2E")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Refit TEST performance (NAS seed {seed}; refit seeds 2021/2022/2023)",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, output / f"refit_test_performance_{seed}.png")


def _plot_test_efficiency(records: dict, output: Path, seed: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 6), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for variant in VARIANTS:
            item = records.get((dataset, variant))
            if not item:
                continue
            for role in ROLES:
                stats = _role_stats(item, role)
                if not stats or "macs_mean" not in stats or "macro_f1_mean" not in stats:
                    continue
                ax.errorbar(
                    stats["macs_mean"], stats["macro_f1_mean"],
                    yerr=stats.get("macro_f1_std", 0.0),
                    color=COLORS[variant], marker=ROLE_MARKERS[role], markersize=8,
                    linestyle="none", capsize=2, alpha=0.9,
                    label=f"{LABELS[variant]} / {ROLE_LABELS[role]}",
                )
        ax.set_xscale("log")
        ax.set_xlabel("MACs (log scale)")
        ax.set_ylabel("TEST Macro-F1 (mean ± std)")
        ax.set_title(dataset)
        _style(ax)
    handles, labels = [], []
    for ax in axes[0]:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.suptitle("Refit TEST performance–cost trade-off", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    _save(fig, output / f"refit_test_f1_macs_{seed}.png")


def _plot_generalization_gap(records: dict, output: Path, seed: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 5.5), squeeze=False)
    positions = np.arange(len(VARIANTS), dtype=float)
    width = 0.24
    role_colors = {"high_f1": "#4C78A8", "knee": "#59A14F", "low_macs": "#F28E2B"}
    for ax, dataset in zip(axes[0], DATASETS):
        for role_index, role in enumerate(ROLES):
            gaps = []
            for variant in VARIANTS:
                item = records.get((dataset, variant))
                stats = _role_stats(item, role) if item else None
                if not stats or not np.isfinite(stats.get("search_macro_f1", np.nan)):
                    gaps.append(np.nan)
                else:
                    gaps.append(stats["search_macro_f1"] - stats["macro_f1_mean"])
            ax.bar(
                positions + (role_index - 1) * width, gaps, width,
                color=role_colors[role], alpha=0.88, label=ROLE_LABELS[role],
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(positions, [LABELS[v] for v in VARIANTS], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Search validation F1 − TEST F1")
        ax.set_title(dataset)
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Search-to-TEST F1 gap (positive = validation optimism)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    _save(fig, output / f"refit_generalization_gap_{seed}.png")


def _plot_refit_summary_table(records: dict, output: Path, seed: int):
    import matplotlib.pyplot as plt

    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            item = records.get((dataset, variant))
            if not item or not item.get("final_rows"):
                rows.append([dataset, LABELS[variant], "Missing", "—", "—", "—"])
                continue
            for role in ROLES:
                stats = _role_stats(item, role)
                if not stats:
                    rows.append([dataset, LABELS[variant], ROLE_LABELS[role], "—", "—", "—"])
                    continue
                rows.append([
                    dataset,
                    LABELS[variant],
                    ROLE_LABELS[role],
                    f"{stats['macro_f1_mean']:.4f} ± {stats.get('macro_f1_std', 0.0):.4f}",
                    f"{stats['accuracy_mean']:.4f} ± {stats.get('accuracy_std', 0.0):.4f}",
                    f"{stats.get('macs_mean', np.nan) / 1e6:.3f}M",
                ])
    fig, ax = plt.subplots(figsize=(14, max(8, 0.31 * len(rows) + 1.6)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Dataset", "Variant", "Role", "TEST Macro-F1", "TEST Accuracy", "MACs"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#34495E")
        elif col == 1:
            variant = next(
                (key for key, label in LABELS.items() if label == cell.get_text().get_text()),
                "baseline",
            )
            cell.set_facecolor(COLORS[variant])
            cell.set_text_props(color="white", weight="bold")
        elif rows[row - 1][2] == "Missing":
            cell.set_facecolor("#FDEDEC")
            cell.set_text_props(color="#922B21")
    ax.set_title(
        f"Refit TEST summary (NAS seed {seed}; mean ± std over refit seeds)",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    _save(fig, output / f"refit_test_summary_table_{seed}.png")


def _write_refit_summary_csv(records: dict, output: Path, seed: int):
    fields = [
        "dataset", "variant", "role", "refit_seed_count", "candidate_hash",
        "test_macro_f1_mean", "test_macro_f1_std", "test_accuracy_mean", "test_accuracy_std",
        "parameters", "macs", "latency_median_ms_mean", "peak_memory_bytes_mean",
        "search_macro_f1", "search_to_test_f1_gap",
    ]
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            item = records.get((dataset, variant))
            for role in ROLES:
                stats = _role_stats(item, role) if item else None
                if not stats:
                    continue
                test_f1 = stats.get("macro_f1_mean", np.nan)
                search_f1 = stats.get("search_macro_f1", np.nan)
                rows.append({
                    "dataset": dataset, "variant": variant, "role": role,
                    "refit_seed_count": stats["n"], "candidate_hash": stats["candidate_hash"],
                    "test_macro_f1_mean": test_f1,
                    "test_macro_f1_std": stats.get("macro_f1_std", ""),
                    "test_accuracy_mean": stats.get("accuracy_mean", ""),
                    "test_accuracy_std": stats.get("accuracy_std", ""),
                    "parameters": stats.get("parameters_mean", ""),
                    "macs": stats.get("macs_mean", ""),
                    "latency_median_ms_mean": stats.get("latency_median_ms_mean", ""),
                    "peak_memory_bytes_mean": stats.get("peak_memory_bytes_mean", ""),
                    "search_macro_f1": search_f1,
                    "search_to_test_f1_gap": search_f1 - test_f1 if np.isfinite(search_f1) and np.isfinite(test_f1) else "",
                })
    path = output / f"ablation_refit_test_summary_{seed}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(records: dict, output: Path):
    fields = [
        "dataset", "variant", "run", "final_hypervolume", "pareto_size",
        "unique_evaluations", "structure_coverage", "tokenizer_coverage",
        "direction_coverage", "mean_genome_distance",
    ]
    rows = []
    for item in records.values():
        rows.append({
            "dataset": item["dataset"],
            "variant": item["variant"],
            "run": str(item["run"]),
            "final_hypervolume": item["final_hv"],
            "pareto_size": item["pareto_size"],
            "unique_evaluations": item["unique_evaluations"],
            "structure_coverage": item["final_structure_coverage"],
            "tokenizer_coverage": item["final_tokenizer_coverage"],
            "direction_coverage": item["final_direction_coverage"],
            "mean_genome_distance": item["final_mean_genome_distance"],
        })
    with (output / "ablation_search_summary_2021.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], VARIANTS.index(row["variant"]))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs", help="Root containing <run-id>/<dataset>/search")
    parser.add_argument("--output", default="results/nas_ablation_2021", help="Output directory")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    input_root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, _ = _load_records(input_root, args.seed)
    _plot_convergence(records, output)
    _plot_diversity(records, output)
    _plot_pareto(records, output)
    _plot_module_summary(records, output)
    _write_summary_csv(records, output)
    _plot_test_performance(records, output, args.seed)
    _plot_test_efficiency(records, output, args.seed)
    _plot_generalization_gap(records, output, args.seed)
    _plot_refit_summary_table(records, output, args.seed)
    _write_refit_summary_csv(records, output, args.seed)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
