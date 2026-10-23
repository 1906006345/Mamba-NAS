"""Plot the five-method revised combination ablation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .plot_ablation_search import COLORS, LABELS, ROLE_LABELS, ROLES, _float, _role_stats, _save, _style

METHODS = ("baseline", "diversity_archive", "local_search", "archive_local_search", "archive_local_search_v2")
METHOD_LABELS = {
    "baseline": "Baseline",
    "diversity_archive": "Diversity archive",
    "local_search": "Local search",
    "archive_local_search": "Original archive+local",
    "archive_local_search_v2": "Revised archive+local",
}
METHOD_COLORS = {
    **COLORS,
    "archive_local_search": "#9467BD",
    "archive_local_search_v2": "#D62728",
}
EXPERIMENTS = (
    ("Heartbeat K-fold", "heartbeat-kfold-{}-seed{}", "Heartbeat"),
    ("SpokenArabicDigits", "spokenarabicdigits-holdout-{}-seed{}", "SpokenArabicDigits"),
    ("PEMS-SF", "pems-sf-holdout-{}-seed{}", "PEMS-SF"),
)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_records(root: Path, seed: int):
    records = {}
    missing = []
    for label, pattern, dataset in EXPERIMENTS:
        for method in METHODS:
            run = root / pattern.format(method, seed) / dataset
            if method == "archive_local_search_v2":
                tagged = root / pattern.format("archive_local_search_v2-slotsfix", seed) / dataset
                if tagged.exists():
                    run = tagged
            search = run / "search"
            if not (search / "generation_diversity.csv").exists():
                missing.append(f"{label}/{method}")
                continue
            generations = sorted(read_csv(search / "generation_diversity.csv"), key=lambda r: int(float(r["generation"])))
            summary = run / "final" / "summary.csv"
            records[(label, method)] = {
                "label": label,
                "method": method,
                "run": run,
                "generations": generations,
                "final": generations[-1],
                "front": read_csv(search / "pareto_front.csv"),
                "summary": read_csv(summary) if summary.exists() else [],
                "reps": json.loads((search / "representatives.json").read_text(encoding="utf-8")),
            }
    if missing:
        print("Warning: missing artifacts:", ", ".join(missing))
    if not records:
        raise FileNotFoundError(f"No revised-combination artifacts found under {root}")
    return records


def save_search_plots(records, output: Path, seed: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), squeeze=False)
    for ax, (label, _, _) in zip(axes[0], EXPERIMENTS):
        for method in METHODS:
            item = records.get((label, method))
            if not item:
                continue
            rows = item["generations"]
            ax.plot([_float(r, "unique_evaluations") for r in rows], [_float(r, "hypervolume") for r in rows],
                    marker="o", markersize=2.5, linewidth=1.8, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(label)
        ax.set_xlabel("Unique evaluations")
        ax.set_ylabel("Hypervolume")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Five-method search convergence (NAS seed {seed})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _save(fig, output / "revised_combo_search_convergence.png")

    metrics = (("hypervolume", "Final HV"), ("structure_coverage", "Structure coverage"), ("direction_coverage", "Direction coverage"), ("mean_genome_distance", "Genome distance"))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    positions = np.arange(len(EXPERIMENTS)); width = 0.16
    for ax, (metric, title) in zip(axes.flat, metrics):
        for idx, method in enumerate(METHODS):
            values = [float(records[(label, method)]["final"].get(metric, 0.0)) for label, _, _ in EXPERIMENTS]
            ax.bar(positions + (idx - 2) * width, values, width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_title(title); ax.set_xticks(positions, [x[0] for x in EXPERIMENTS], rotation=25, ha="right"); _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Five-method search quality and diversity", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "revised_combo_search_metrics.png")


def save_test_plots(records, output: Path, seed: int):
    import matplotlib.pyplot as plt

    roles = ("high_f1", "knee", "low_macs")
    role_colors = {"high_f1": "#4C78A8", "knee": "#59A14F", "low_macs": "#F28E2B"}
    positions = np.arange(len(METHODS)); width = 0.22
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), squeeze=False)
    for ax, (label, _, _) in zip(axes[0], EXPERIMENTS):
        for idx, role in enumerate(roles):
            means, stds = [], []
            for method in METHODS:
                stats = _role_stats(records[(label, method)], role)
                means.append(stats.get("macro_f1_mean", np.nan) if stats else np.nan)
                stds.append(stats.get("macro_f1_std", 0.0) if stats else 0.0)
            ax.bar(positions + (idx - 1) * width, means, width, yerr=stds, capsize=2.5, color=role_colors[role], label=ROLE_LABELS[role])
        ax.set_title(label); ax.set_ylabel("TEST Macro-F1 mean ± std"); ax.set_xticks(positions, [METHOD_LABELS[m] for m in METHODS], rotation=35, ha="right", fontsize=8); _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Refit TEST performance (NAS seed {seed})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94)); _save(fig, output / "revised_combo_test_performance.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), squeeze=False)
    markers = {"high_f1": "*", "knee": "D", "low_macs": "P"}
    for ax, (label, _, _) in zip(axes[0], EXPERIMENTS):
        for method in METHODS:
            for role in roles:
                stats = _role_stats(records[(label, method)], role)
                if not stats: continue
                ax.errorbar(stats["macs_mean"], stats["macro_f1_mean"], yerr=stats.get("macro_f1_std", 0.0), color=METHOD_COLORS[method], marker=markers[role], markersize=8, linestyle="none", capsize=2, label=f"{METHOD_LABELS[method]} / {ROLE_LABELS[role]}")
        ax.set_xscale("log"); ax.set_title(label); ax.set_xlabel("MACs (log)"); ax.set_ylabel("TEST Macro-F1"); _style(ax)
    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels(); handles.extend(h); labels.extend(l)
    unique = dict(zip(labels, handles)); fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.suptitle("Five-method TEST performance–cost trade-off", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.15, 1, 0.94)); _save(fig, output / "revised_combo_test_f1_macs.png")


def save_search_diagnostics(records, output: Path):
    import matplotlib.pyplot as plt

    metrics = (
        ("cumulative_local_search_slot_fill_rate", "Local slot fill rate"),
        ("local_search_cumulative_positive_f1_rate", "Positive ΔF1 rate"),
        ("local_search_cumulative_pareto_entry_rate", "Local Pareto entry rate"),
        ("archive_source_pareto_rate", "Archive-source Pareto rate"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    positions = np.arange(len(EXPERIMENTS)); width = 0.16
    for ax, (metric, title) in zip(axes.flat, metrics):
        for idx, method in enumerate(METHODS):
            values = [
                _float(records[(label, method)]["final"], metric, 0.0)
                for label, _, _ in EXPERIMENTS
            ]
            ax.bar(positions + (idx - 2) * width, values, width,
                   color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(title)
        ax.set_xticks(positions, [item[0] for item in EXPERIMENTS], rotation=25, ha="right")
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Archive/local-search diagnostics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "revised_combo_search_diagnostics.png")


def write_csv(records, output: Path):
    fields = ["experiment", "method", "final_hv", "structure_coverage", "direction_coverage", "mean_genome_distance", "pareto_size", "macro_structure_coverage", "micro_structure_coverage", "requested_local_slots", "actual_local_children", "fallback_local_children", "fallback_variation_children", "local_search_slot_fill_rate", "local_search_cumulative_positive_f1_rate", "local_search_cumulative_pareto_entry_rate", "local_search_cumulative_mean_hv_contribution", "archive_source_pareto_rate", "high_f1_test_mean", "high_f1_test_std", "high_f1_macs"]
    rows = []
    for (label, method), item in records.items():
        stats = _role_stats(item, "high_f1")
        final = item["final"]
        rows.append({"experiment": label, "method": method, "final_hv": final["hypervolume"], "structure_coverage": final["structure_coverage"], "direction_coverage": final["direction_coverage"], "mean_genome_distance": final["mean_genome_distance"], "pareto_size": len(item["front"]), "macro_structure_coverage": final.get("macro_structure_coverage", ""), "micro_structure_coverage": final.get("micro_structure_coverage", ""), "requested_local_slots": final.get("cumulative_requested_local_slots", final.get("requested_local_slots", "")), "actual_local_children": final.get("cumulative_actual_local_children", final.get("actual_local_children", "")), "fallback_local_children": final.get("cumulative_fallback_local_children", final.get("fallback_local_children", "")), "fallback_variation_children": final.get("cumulative_fallback_variation_children", final.get("fallback_variation_children", "")), "local_search_slot_fill_rate": final.get("cumulative_local_search_slot_fill_rate", final.get("local_search_slot_fill_rate", "")), "local_search_cumulative_positive_f1_rate": final.get("local_search_cumulative_positive_f1_rate", ""), "local_search_cumulative_pareto_entry_rate": final.get("local_search_cumulative_pareto_entry_rate", ""), "local_search_cumulative_mean_hv_contribution": final.get("local_search_cumulative_mean_hv_contribution", ""), "archive_source_pareto_rate": final.get("archive_source_pareto_rate", ""), "high_f1_test_mean": stats.get("macro_f1_mean", "") if stats else "", "high_f1_test_std": stats.get("macro_f1_std", "") if stats else "", "high_f1_macs": stats.get("macs_mean", "") if stats else ""})
    with (output / "revised_combo_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs"); parser.add_argument("--output", default="results/revised_combo_ablation"); parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args(); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.input).resolve(), args.seed); save_search_plots(records, output, args.seed); save_search_diagnostics(records, output); save_test_plots(records, output, args.seed); write_csv(records, output); print(output)


if __name__ == "__main__":
    raise SystemExit(main())
