"""Create a results-style table for baseline models vs archive_local_search knee.

The comparison uses the three datasets for which PatchTST, TimesNet and the
new multi-layer Mamba results are present in ``results/``. The NAS column is
the original (unmodified) ``archive_local_search`` holdout knee architecture
from NAS seed 2021, aggregated over its three refit seeds.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path


DATASETS = ("Heartbeat", "SpokenArabicDigits", "PEMS-SF")
MODEL_ORDER = ("PatchTST", "TimesNet", "MambaMultiLayer", "archive_local_search")
MODEL_LABELS = {model: model for model in MODEL_ORDER}


def _parse_result(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    def value(name: str) -> float:
        match = re.search(rf"^{re.escape(name)}:\s*([0-9.eE+-]+)", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing {name} in {path}")
        return float(match.group(1))
    return {
        "accuracy": value("accuracy"),
        "macro_f1": value("macro_f1"),
        "parameters": value("parameters"),
        "macs": value("macs"),
    }


def _model_result(dataset: str, model: str) -> dict:
    if model == "MambaMultiLayer":
        pattern = f"results/classification_{dataset}_Mamba_*MambaMultiLayer*_fair_s*/result_classification.txt"
    else:
        pattern = f"results/classification_{dataset}_{model}_*_fair_s*/result_classification.txt"
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No {model} results found for {dataset}: {pattern}")
    rows = [_parse_result(path) for path in paths]
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = sum(values) / len(values)
        result[f"{key}_std"] = (
            sum((value - result[key]) ** 2 for value in values) / (len(values) - 1)
        ) ** 0.5 if len(values) > 1 else 0.0
    result["n"] = len(rows)
    return result


def _archive_knee_result(dataset: str) -> dict:
    prefix = {"Heartbeat": "heartbeat", "SpokenArabicDigits": "spokenarabicdigits", "PEMS-SF": "pems-sf"}[dataset]
    path = Path("runs") / f"{prefix}-holdout-archive_local_search-seed2021" / dataset / "final" / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [row for row in csv.DictReader(path.open(encoding="utf-8")) if row["role"] == "knee"]
    if not rows:
        raise ValueError(f"No knee rows in {path}")
    numeric = {"accuracy": [], "macro_f1": [], "parameters": [], "macs": []}
    for row in rows:
        for key in numeric:
            numeric[key].append(float(row[key]))
    result = {}
    for key, values in numeric.items():
        result[key] = sum(values) / len(values)
        result[f"{key}_std"] = (
            sum((value - result[key]) ** 2 for value in values) / (len(values) - 1)
        ) ** 0.5 if len(values) > 1 else 0.0
    result["n"] = len(rows)
    return result


def _format_score(row: dict) -> str:
    return f"{row['accuracy']:.4f} / {row['macro_f1']:.4f}"


def _format_score_with_std(row: dict) -> str:
    return f"{row['accuracy']:.4f} / {row['macro_f1']:.4f}"


def _format_cost(row: dict) -> str:
    return f"{row['parameters'] / 1_000_000:.3f}M / {row['macs'] / 1_000_000:.3f}M"


def make_table(output: str | Path = "results/model_archive_comparison.png") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {
        dataset: {
            **{model: _model_result(dataset, model) for model in MODEL_ORDER if model != "archive_local_search"},
            "archive_local_search": _archive_knee_result(dataset),
        }
        for dataset in DATASETS
    }
    accuracy_rows = [
        [dataset] + [_format_score(values[dataset][model]) for model in MODEL_ORDER]
        for dataset in DATASETS
    ]
    cost_rows = [
        [dataset] + [_format_cost(values[dataset][model]) for model in MODEL_ORDER]
        for dataset in DATASETS
    ]
    headers_score = ["Dataset"] + [f"{MODEL_LABELS[model]}\nAccuracy / Macro-F1" for model in MODEL_ORDER]
    headers_cost = ["Dataset"] + [f"{MODEL_LABELS[model]}\nParameters / MACs" for model in MODEL_ORDER]

    fig, axes = plt.subplots(
        2, 1, figsize=(18.5, 7.6), dpi=160,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.42},
    )
    column_widths = [0.20, 0.16, 0.16, 0.19, 0.20]
    for ax, rows, headers, title in zip(
        axes, [accuracy_rows, cost_rows], [headers_score, headers_cost],
        ["Accuracy / Macro-F1", "Trainable Parameters / Forward MACs"],
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        table = ax.table(
            cellText=rows, colLabels=headers, colWidths=column_widths,
            cellLoc="center", colLoc="center", loc="center",
            bbox=[0.015, 0.1, 0.97, 0.75],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 1.55)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#C7D0D9")
            cell.set_linewidth(0.8)
            if row == 0:
                cell.set_facecolor("#23384D")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
                cell.get_text().set_fontsize(9.5)
            else:
                cell.set_facecolor("#F7F9FB" if row % 2 else "#EDF3F7")
                if col == 0:
                    cell.set_facecolor("#DCE8F2")
                    cell.get_text().set_weight("bold")
                elif col == len(MODEL_ORDER):
                    cell.set_facecolor("#FBE8E8" if row % 2 else "#F6DDDD")
                    cell.get_text().set_weight("bold")

    fig.suptitle(
        "PatchTST / TimesNet / MambaMultiLayer vs archive_local_search knee",
        fontsize=18, fontweight="bold", y=0.98,
    )
    fig.text(
        0.015, 0.02,
        "Baseline model values: mean over result seeds 2021/2022/2023. "
        "archive_local_search: original holdout NAS seed 2021 knee, mean over refit seeds 2021/2022/2023. "
        "Units in the performance table are millions (M).",
        fontsize=9, color="#45515C",
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/model_archive_comparison.png")
    args = parser.parse_args()
    print(make_table(args.output))
