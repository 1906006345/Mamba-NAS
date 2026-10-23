"""Generate a focused weekly report for baseline and single mechanisms.

Included variants are deliberately limited to:
``baseline``, ``dual_crowding``, ``diversity_archive`` and ``local_search``.
The script combines search artifacts and refit summaries, without using the
excluded ``dual_space`` or ``full`` variants.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .plot_ablation_search import COLORS, LABELS, ROLES, ROLE_LABELS, _float, _role_stats, _save, _style


VARIANTS = ("baseline", "dual_crowding", "diversity_archive", "local_search")
DATASETS = (
    ("Heartbeat holdout", "heartbeat-holdout-{}-seed2021", "Heartbeat"),
    ("Heartbeat K-fold", "heartbeat-kfold-{}-seed2021", "Heartbeat"),
    ("SpokenArabicDigits", "spokenarabicdigits-holdout-{}-seed2021", "SpokenArabicDigits"),
    ("UWaveGestureLibrary", "uwavegesturelibrary-holdout-{}-seed2021", "UWaveGestureLibrary"),
    ("PEMS-SF", "pems-sf-holdout-{}-seed2021", "PEMS-SF"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_records(input_root: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    missing = []
    for label, pattern, dataset in DATASETS:
        for variant in VARIANTS:
            run = input_root / pattern.format(variant) / dataset
            search = run / "search"
            if not (search / "generation_diversity.csv").exists():
                missing.append(f"{label}/{variant}")
                continue
            generations = _read_csv(search / "generation_diversity.csv")
            generations.sort(key=lambda row: int(float(row["generation"])))
            evaluations = [
                row for row in _read_csv(search / "evaluations.csv")
                if row.get("status", "completed") == "completed"
            ]
            pareto = _read_csv(search / "pareto_front.csv")
            representatives_path = search / "representatives.json"
            representatives = json.loads(representatives_path.read_text(encoding="utf-8"))
            summary_path = run / "final" / "summary.csv"
            final_rows = _read_csv(summary_path) if summary_path.exists() else []
            local_path = search / "local_search.csv"
            local_rows = _read_csv(local_path) if local_path.exists() else []
            final = generations[-1]
            records[(label, variant)] = {
                "label": label,
                "dataset": dataset,
                "variant": variant,
                "run": run,
                "generations": generations,
                "evaluations": evaluations,
                "pareto": pareto,
                "representatives": representatives,
                "final_rows": final_rows,
                "local_rows": local_rows,
                "final_hv": _float(final, "hypervolume"),
                "structure_coverage": _float(final, "structure_coverage"),
                "direction_coverage": _float(final, "direction_coverage"),
                "mean_genome_distance": _float(final, "mean_genome_distance"),
                "pareto_size": len(pareto),
            }
    if missing:
        print("Warning: missing artifacts:", ", ".join(missing))
    if not records:
        raise FileNotFoundError(f"No single-method artifacts found under {input_root}")
    return records


def _plot_search_convergence(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for ax, (label, _, _) in zip(axes.flat, DATASETS):
        for variant in VARIANTS:
            item = records.get((label, variant))
            if not item:
                continue
            rows = item["generations"]
            ax.plot(
                [_float(row, "unique_evaluations") for row in rows],
                [_float(row, "hypervolume") for row in rows],
                color=COLORS[variant], marker="o", markersize=2.5,
                linewidth=1.8, label=LABELS[variant],
            )
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Unique candidate evaluations")
        ax.set_ylabel("Hypervolume")
        _style(ax)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Single-method search convergence", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    _save(fig, output / "single_methods_search_convergence.png")


def _plot_search_metrics(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("final_hv", "Final Hypervolume"),
        ("structure_coverage", "Structure coverage"),
        ("direction_coverage", "Direction coverage"),
        ("mean_genome_distance", "Mean weighted Genome distance"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), squeeze=False)
    positions = np.arange(len(DATASETS))
    width = 0.19
    for ax, (metric, title) in zip(axes.flat, metrics):
        for index, variant in enumerate(VARIANTS):
            values = [records[(label, variant)][metric] for label, _, _ in DATASETS]
            bars = ax.bar(
                positions + (index - 1.5) * width, values, width,
                color=COLORS[variant], alpha=0.88, label=LABELS[variant],
            )
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_title(title)
        ax.set_xticks(positions, [label for label, _, _ in DATASETS], rotation=25, ha="right")
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Single-method search metrics", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output / "single_methods_search_metrics.png")


def _plot_test_performance(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    positions = np.arange(len(VARIANTS))
    width = 0.25
    role_colors = {"high_f1": "#4C78A8", "knee": "#59A14F", "low_macs": "#F28E2B"}
    for ax, (label, _, _) in zip(axes.flat, DATASETS):
        all_values = []
        for role_index, role in enumerate(ROLES):
            means, stds = [], []
            for variant in VARIANTS:
                stats = _role_stats(records[(label, variant)], role)
                means.append(stats["macro_f1_mean"])
                stds.append(stats.get("macro_f1_std", 0.0))
            all_values.extend(means)
            ax.bar(positions + (role_index - 1) * width, means, width, yerr=stds,
                   capsize=2.5, color=role_colors[role], alpha=0.88,
                   label=ROLE_LABELS[role])
        margin = max(0.01, (max(all_values) - min(all_values)) * 0.30)
        ax.set_ylim(max(0, min(all_values) - margin), min(1.01, max(all_values) + margin))
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("TEST Macro-F1 (mean ± std)")
        ax.set_xticks(positions, [LABELS[v] for v in VARIANTS], rotation=25, ha="right", fontsize=8)
        _style(ax)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Single-method refit TEST Macro-F1", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    _save(fig, output / "single_methods_test_performance.png")


def _plot_test_tradeoff(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    markers = {"high_f1": "*", "knee": "D", "low_macs": "P"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    for ax, (label, _, _) in zip(axes.flat, DATASETS):
        for variant in VARIANTS:
            for role in ROLES:
                stats = _role_stats(records[(label, variant)], role)
                ax.errorbar(stats["macs_mean"], stats["macro_f1_mean"],
                            yerr=stats.get("macro_f1_std", 0.0),
                            color=COLORS[variant], marker=markers[role], markersize=8,
                            linestyle="none", capsize=2, alpha=0.9,
                            label=f"{LABELS[variant]} / {ROLE_LABELS[role]}")
        ax.set_xscale("log")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("MACs (log scale)")
        ax.set_ylabel("TEST Macro-F1")
        _style(ax)
    axes.flat[-1].axis("off")
    handles, labels = [], []
    for ax in axes.flat[:-1]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.suptitle("Single-method TEST performance–cost trade-off", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    _save(fig, output / "single_methods_test_f1_macs.png")


def _local_search_stats(records: dict) -> list[dict]:
    rows = []
    for (label, variant), item in records.items():
        if variant != "local_search":
            continue
        local_rows = item["local_rows"]
        deltas = []
        for row in local_rows:
            parent = _float(row, "parent_macro_f1")
            child = _float(row, "child_macro_f1")
            if np.isfinite(parent) and np.isfinite(child):
                deltas.append(child - parent)
        counts: dict[str, int] = {}
        for row in local_rows:
            counts[row.get("changed_gene", "unknown")] = counts.get(row.get("changed_gene", "unknown"), 0) + 1
        rows.append({
            "experiment": label,
            "local_candidates": len(local_rows),
            "mean_delta_f1": float(np.mean(deltas)) if deltas else np.nan,
            "positive_delta_rate": float(np.mean(np.asarray(deltas) > 0)) if deltas else np.nan,
            "gene_counts": counts,
        })
    return rows


def _plot_local_search(records: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    stats = _local_search_stats(records)
    labels = [row["experiment"] for row in stats]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    candidates = [row["local_candidates"] for row in stats]
    rates = [row["positive_delta_rate"] * 100 if np.isfinite(row["positive_delta_rate"]) else 0 for row in stats]
    axes[0].bar(positions, candidates, color="#F28E2B", alpha=0.88)
    axes[0].set_title("Evaluated local-search children")
    axes[0].set_ylabel("Candidate count")
    axes[1].bar(positions, rates, color="#59A14F", alpha=0.88)
    axes[1].set_title("Local child with positive ΔMacro-F1")
    axes[1].set_ylabel("Percentage (%)")
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=25, ha="right")
        _style(ax)
    fig.suptitle("Pareto Local Search diagnostics", fontsize=17, fontweight="bold")
    fig.tight_layout()
    _save(fig, output / "local_search_diagnostics.png")


def _write_csv(records: dict, output: Path) -> None:
    fields = ["experiment", "variant", "final_hypervolume", "structure_coverage",
              "direction_coverage", "mean_genome_distance", "pareto_size",
              "high_f1_test_mean", "high_f1_test_std", "high_f1_macs"]
    rows = []
    for (label, variant), item in records.items():
        stats = _role_stats(item, "high_f1")
        rows.append({
            "experiment": label,
            "variant": variant,
            "final_hypervolume": item["final_hv"],
            "structure_coverage": item["structure_coverage"],
            "direction_coverage": item["direction_coverage"],
            "mean_genome_distance": item["mean_genome_distance"],
            "pareto_size": item["pareto_size"],
            "high_f1_test_mean": stats.get("macro_f1_mean", "") if stats else "",
            "high_f1_test_std": stats.get("macro_f1_std", "") if stats else "",
            "high_f1_macs": stats.get("macs_mean", "") if stats else "",
        })
    with (output / "single_methods_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(records: dict, output: Path) -> None:
    high_f1_rows = []
    for label, _, _ in DATASETS:
        values = []
        for variant in VARIANTS:
            item = records[(label, variant)]
            stats = _role_stats(item, "high_f1")
            if stats:
                values.append(
                    f"{LABELS[variant]}: {stats['macro_f1_mean']:.4f} ± {stats.get('macro_f1_std', 0.0):.4f}"
                )
            else:
                values.append("—")
        high_f1_rows.append(f"| {label} | " + " | ".join(values) + " |")
    local_rows = _local_search_stats(records)
    local_table = []
    for row in local_rows:
        rate = row["positive_delta_rate"] * 100 if np.isfinite(row["positive_delta_rate"]) else float("nan")
        local_table.append(
            f"| {row['experiment']} | {row['local_candidates']} | {rate:.1f}% |"
        )
    lines = [
        "# Mamba-NAS 单模块消融周报",
        "",
        "## 实验范围",
        "",
        "本周只比较 Baseline、Decision Crowding、Diversity Archive 和 Pareto Local Search，排除 Dual Space 与 Full。实验包含 Heartbeat holdout、Heartbeat 5-fold K-fold、SpokenArabicDigits holdout、UWaveGestureLibrary holdout 和 PEMS-SF holdout。",
        "",
        "## Local Search 设计",
        "",
        "Pareto Local Search 不使用代理模型，也不增加总候选预算。每代从当前全局 Pareto 架构、结构档案 Elite（若启用）和当前种群中选取种子，枚举只改变一个基因的合法一阶邻居。邻居优先顺序为 tokenizer、direction、pooling、num_blocks、d_model、d_state、d_conv、expand。默认每代占 offspring 的 30%，重复候选和 cache 中已有候选会被跳过，剩余槽位由普通交叉/变异和随机移民补齐。",
        "",
        "它的作用是补充 NSGA-II 的随机全局探索：NSGA-II 负责发现不同区域，Local Search 负责系统开发高价值架构周围的局部邻域。当前实现是每代一次一阶邻域枚举，不是递归爬山；完整候选仍由原 evaluator 训练，holdout 和 K-fold 都兼容。记录文件为 `search/local_search.csv`。",
        "",
        "## 图表",
        "",
        "![搜索收敛](single_methods_search_convergence.png)",
        "",
        "![搜索指标](single_methods_search_metrics.png)",
        "",
        "![TEST Macro-F1](single_methods_test_performance.png)",
        "",
        "![TEST F1-MACs](single_methods_test_f1_macs.png)",
        "",
        "![Local Search 诊断](local_search_diagnostics.png)",
        "",
        "## high-F1 TEST 对比",
        "",
        "以下为三个 refit seeds 的 TEST Macro-F1 均值 ± 标准差：",
        "",
        "| 实验 | Baseline | Decision Crowding | Diversity Archive | Local Search |",
        "|---|---:|---:|---:|---:|",
        *high_f1_rows,
        "",
        "## Local Search 子代诊断",
        "",
        "| 实验 | 局部候选数 | 子代 F1 高于父代比例 |",
        "|---|---:|---:|",
        *local_table,
        "",
        "## 结果结论",
        "",
        "- Heartbeat holdout 中，Decision Crowding 的搜索 HV 最高，但 Local Search 的 TEST high-F1 最高；Heartbeat K-fold 中，在本周纳入的三个单模块里，Local Search 的 TEST high-F1 仍然最高，说明局部搜索对 Heartbeat 的泛化最稳定。完整组合 Full 不属于本周单模块图表。",
        "- SpokenArabicDigits 中，Diversity Archive 的结构覆盖和 Genome 距离最好，Decision Crowding 的 TEST high-F1 最好；Local Search 虽然搜索 HV 最高，但 TEST high-F1 没有提升，说明局部邻域开发可能过拟合 holdout。",
        "- UWaveGestureLibrary 中，Dual Space 才带来明显 TEST high-F1 提升；单独 Local Search 与 Baseline 基本相同，说明该数据集的有效结构变化可能不是单基因邻域能够发现的。",
        "- PEMS-SF 中，Decision Crowding 的 TEST high-F1 最好且 MACs 较低；Local Search 的 TEST high-F1 最低，说明局部搜索在该数据集上没有带来泛化收益。",
        "- Diversity Archive 的主要价值是结构探索和结构保留，而不是直接提高 TEST F1；Decision Crowding 更偏向目标收敛；Local Search 对 Heartbeat 最有帮助，但具有明显数据集依赖。",
        "",
        "## 当前结论",
        "",
        "单模块结果支持三种机制的不同作用，而不支持某一种机制在所有数据集上普遍最优。建议论文中将 Diversity Archive 定位为结构多样性机制，将 Decision Crowding 定位为目标空间收敛增强，将 Local Search 定位为数据集依赖的局部开发机制；Heartbeat 应优先使用 K-fold，避免根据单次 holdout HV 判断方法优劣。",
    ]
    (output / "WEEKLY_REPORT_SINGLE_METHODS_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs")
    parser.add_argument("--output", default="results/weekly_single_methods")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(args.input).expanduser().resolve())
    _plot_search_convergence(records, output)
    _plot_search_metrics(records, output)
    _plot_test_performance(records, output)
    _plot_test_tradeoff(records, output)
    _plot_local_search(records, output)
    _write_csv(records, output)
    _write_report(records, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
