"""Generate ``results/accuracy_f1_comparison.png`` from editable values.

Edit only ``RESULTS`` below when replacing experiment numbers, then run:

    python -m mamba_nas.update_comparison_png

The MambaTS entries below are the mean over classification seeds 2021/2022/2023.
No CSV, checkpoint, or run directory is read by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Editable data area
# ---------------------------------------------------------------------------
# Each record must contain accuracy, macro_f1, parameters, and macs.
# Integers may use underscores (for example, 104_562) for readability.

DATASET_ORDER = ["Heartbeat", "SpokenArabicDigits", "PEMS-SF"]
MODEL_ORDER = ["PatchTST", "TimesNet", "Mamba", "MambaTS", "TSCMamba", "Mamba-NAS-base"]
MODEL_LABELS = {
    "PatchTST": "PatchTST",
    "TimesNet": "TimesNet",
    "Mamba": "Mamba",
    "MambaTS": "MambaTS",
    "TSCMamba": "TSCMamba",
    "Mamba-NAS-base": "Mamba-NAS-base",
}

RESULTS = {
    "Heartbeat": {
        "PatchTST": {"accuracy": 0.6732, "macro_f1": 0.5288,
                     "parameters": 0.105, "macs": 26.889},
        "TimesNet": {"accuracy": 0.7317, "macro_f1": 0.5966,
                     "parameters": 0.895, "macs": 512.844},
        "Mamba": {"accuracy": 0.7220, "macro_f1": 0.4193,
                 "parameters": 0.019, "macs": 4.859},
        # Mean over seeds 2021/2022/2023:
        # accuracy=0.726829, macro_f1=0.457800, parameters=0.010402M,
        # MACs=24.306976M.
        "MambaTS": {"accuracy": 0.7268, "macro_f1": 0.4578,
                    "parameters": 0.010, "macs": 24.307},
        # Mean over seeds 2021/2022/2023, e_layers=3.
        "TSCMamba": {"accuracy": 0.7024, "macro_f1": 0.6196,
                     "parameters": 0.364, "macs": 44.189},
        "Mamba-NAS-base": {"accuracy": 0.7528, "macro_f1": 0.6836,
                 "parameters": 0.064, "macs": 3.444},
    },
    "SpokenArabicDigits": {
        "PatchTST": {"accuracy": 0.9650, "macro_f1": 0.9650,
                     "parameters": 0.059, "macs": 1.516},
        "TimesNet": {"accuracy": 0.9823, "macro_f1": 0.9823,
                     "parameters": 1.203, "macs": 270.648},
        "Mamba": {"accuracy": 0.9777, "macro_f1": 0.9777,
                 "parameters": 0.028, "macs": 1.995},
        # Mean over seeds 2021/2022/2023.
        "MambaTS": {"accuracy": 0.9341, "macro_f1": 0.9341,
                    "parameters": 0.021, "macs": 2.264},
        # Mean over seeds 2021/2022/2023, e_layers=2.
        "TSCMamba": {"accuracy": 0.9723, "macro_f1": 0.9723,
                     "parameters": 0.281, "macs": 7.294},
        "Mamba-NAS-base": {"accuracy": 0.9780, "macro_f1": 0.9780,
                 "parameters": 0.040, "macs": 0.519},
    },
    "PEMS-SF": {
        "PatchTST": {"accuracy": 0.5992, "macro_f1": 0.5978,
                     "parameters": 7.918, "macs": 1983.843},
        "TimesNet": {"accuracy": 0.8150, "macro_f1": 0.8102,
                     "parameters": 14.312, "macs": 6073.012},
        "Mamba": {"accuracy": 0.6879, "macro_f1": 0.6675,
                 "parameters": 0.554, "macs": 45.298},
        # Mean over seeds 2021/2022/2023.
        # Mean over seeds 2021/2022/2023, e_layers=3 for both Mamba models.
        "MambaTS": {"accuracy": 0.7034, "macro_f1": 0.6800,
                    "parameters": 0.100, "macs": 1475.293},
        "TSCMamba": {"accuracy": 0.7958, "macro_f1": 0.7852,
                     "parameters": 21.019, "macs": 3672.308},
        "Mamba-NAS-base": {"accuracy": 0.8150, "macro_f1": 0.8170,
                 "parameters": 1.017, "macs": 9.159},
    },
}


# ---------------------------------------------------------------------------
# Rendering code; normally no changes are needed below this line.
# ---------------------------------------------------------------------------

def _validate_results() -> None:
    required = {"accuracy", "macro_f1", "parameters", "macs"}
    for dataset in DATASET_ORDER:
        if dataset not in RESULTS:
            raise ValueError(f"RESULTS is missing dataset {dataset!r}")
        for model in MODEL_ORDER:
            if model not in RESULTS[dataset]:
                raise ValueError(f"RESULTS[{dataset!r}] is missing model {model!r}")
            missing = required - RESULTS[dataset][model].keys()
            if missing:
                raise ValueError(
                    f"RESULTS[{dataset!r}][{model!r}] is missing {sorted(missing)}"
                )


def _format_metrics(row: dict) -> str:
    if row["accuracy"] is None or row["macro_f1"] is None:
        return "N/A"
    return f"{float(row['accuracy']):.4f} / {float(row['macro_f1']):.4f}"


def _format_performance(row: dict) -> str:
    if row["parameters"] is None or row["macs"] is None:
        return "N/A"
    parameters = float(row["parameters"]) 
    macs = float(row["macs"]) 
    return f"{parameters:.3f}M / {macs:.3f}M"


def _style_table(table) -> None:
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
            elif col >= 3:
                cell.set_facecolor("#EAF4EA" if row % 2 else "#E2F0E2")


def make_table(output: str | Path = "results/accuracy_f1_comparison.png") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _validate_results()
    accuracy_rows = [
        [dataset]
        + [_format_metrics(RESULTS[dataset][model]) for model in MODEL_ORDER]
        for dataset in DATASET_ORDER
    ]
    performance_rows = [
        [dataset]
        + [_format_performance(RESULTS[dataset][model]) for model in MODEL_ORDER]
        for dataset in DATASET_ORDER
    ]
    metric_headers = ["Dataset"] + [
        f"{MODEL_LABELS[model]}\nAccuracy / Macro-F1" for model in MODEL_ORDER
    ]
    performance_headers = ["Dataset"] + [
        f"{MODEL_LABELS[model]}\nParameters / MACs" for model in MODEL_ORDER
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(21.0, 7.6),
        dpi=160,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.42},
    )
    column_widths = [0.13, 0.115, 0.115, 0.14, 0.14, 0.14, 0.14, 0.14]
    for ax, rows, headers, title in zip(
        axes,
        [accuracy_rows, performance_rows],
        [metric_headers, performance_headers],
        ["Accuracy / Macro-F1", "Trainable Parameters / Forward MACs"],
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            colWidths=column_widths,
            cellLoc="center",
            colLoc="center",
            loc="center",
            bbox=[0.015, 0.1, 0.97, 0.75],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 1.55)
        _style_table(table)

    fig.suptitle(
        "Baseline Models vs Mamba-NAS on Three UEA Datasets",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.015,
        0.02,
        "Accuracy table: Accuracy / Macro-F1. Performance table: Parameters / MACs, units = millions (M).",
        fontsize=9.5,
        color="#45515C",
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/accuracy_f1_comparison.png",
        help="Destination PNG path",
    )
    args = parser.parse_args()
    print(make_table(args.output))
