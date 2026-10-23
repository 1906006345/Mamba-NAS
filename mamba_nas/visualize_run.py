"""Visualize saved Mamba-NAS run artifacts.

Examples
--------
Render one artifact next to the source file::

    python -m mamba_nas.visualize_run \
        --input runs/heartbeat-paper-kfold/Heartbeat/search/evaluations.csv

Render a whole dataset run overview::

    python -m mamba_nas.visualize_run \
        --input runs/heartbeat-paper-kfold/Heartbeat \
        --output heartbeat_overview.png

The input path is the only experiment-specific argument. The script reads the
selected file/directory and writes a PNG; it does not print raw experiment
parameters. Matplotlib, NumPy and the Python standard library are sufficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROLE_ORDER = ("high_f1", "high_accuracy", "knee", "low_macs", "low_cost")
DATASET_NAMES = {"Heartbeat", "SpokenArabicDigits", "UWaveGestureLibrary"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key)
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _output_path(input_path: Path, requested: str | Path | None) -> Path:
    if requested is not None:
        output = Path(requested)
    elif input_path.is_file():
        output = input_path.with_suffix(".png")
    else:
        output = input_path / "run_overview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _save(fig, output: Path) -> Path:
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return output


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_pareto_front(ax, x_values, y_values, label: str = "Pareto front") -> None:
    """Draw actual Pareto points plus a shape-preserving smooth visual guide."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    # A Pareto front should not contain two useful points at exactly the same
    # MACs, but aggregate defensively when old artifacts do.
    unique_x = np.unique(x)
    unique_y = np.asarray([y[x == value].max() for value in unique_x])
    if len(unique_x) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator

            grid = np.linspace(np.log10(unique_x[0]), np.log10(unique_x[-1]), 240)
            smooth = PchipInterpolator(np.log10(unique_x), unique_y)(grid)
            ax.plot(10**grid, smooth, color="#D1495B", linewidth=2.2, alpha=0.88, label=label, zorder=3)
        except ImportError:
            ax.plot(unique_x, unique_y, color="#D1495B", linewidth=1.8, alpha=0.88, label=label, zorder=3)
    else:
        ax.plot(unique_x, unique_y, color="#D1495B", linewidth=1.8, alpha=0.88, label=label, zorder=3)
    ax.scatter(unique_x, unique_y, s=56, color="#D1495B", edgecolors="white", linewidths=0.7, zorder=4)


def _plot_evaluations(path: Path, output: Path):
    import matplotlib.pyplot as plt

    rows = [r for r in _read_csv(path) if r.get("status", "completed") == "completed"]
    if not rows:
        raise ValueError(f"No completed candidates found in {path}")
    x = np.asarray([_float(r, "macs") for r in rows], dtype=float)
    y = np.asarray([_float(r, "macro_f1") for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.scatter(
        x,
        y,
        s=25,
        alpha=0.35,
        color="#9E9E9E",
        edgecolors="none",
        label="all evaluated candidates",
    )
    ax.set_xscale("log")
    pareto_path = path.with_name("pareto_front.csv")
    if pareto_path.exists() and pareto_path != path:
        pareto = _read_csv(pareto_path)
        px = [_float(r, "macs") for r in pareto]
        py = [_float(r, "macro_f1") for r in pareto]
        _draw_pareto_front(ax, px, py)
    representatives_path = path.with_name("representatives.json")
    if representatives_path.exists():
        representatives = json.loads(representatives_path.read_text(encoding="utf-8"))
        styles = {
            "high_f1": ("#1F77B4", "*"),
            "high_accuracy": ("#1F77B4", "*"),
            "knee": ("#2CA02C", "D"),
            "low_macs": ("#FF7F0E", "P"),
            "low_cost": ("#FF7F0E", "P"),
        }
        row_by_hash = {r.get("candidate_hash"): r for r in rows}
        plotted_hashes = set()
        for role, representative in representatives.items():
            candidate_hash = representative.get("candidate_hash")
            row = row_by_hash.get(candidate_hash, representative)
            xx, yy = _float(row, "macs"), _float(row, "macro_f1")
            if xx is None or yy is None:
                continue
            color, marker = styles.get(role, ("#9467BD", "o"))
            label = role if candidate_hash not in plotted_hashes else None
            ax.scatter([xx], [yy], s=170, color=color, marker=marker, edgecolors="black", linewidths=0.9, label=label, zorder=6)
            ax.annotate(role, (xx, yy), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=color, weight="bold")
            plotted_hashes.add(candidate_hash)
    ax.set_xlabel("MACs (log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Candidate search and Pareto front: {path.parent.parent.name}")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    return _save(fig, output)


def _plot_pareto(path: Path, output: Path):
    import matplotlib.pyplot as plt

    rows = _read_csv(path)
    if not rows or "macs" not in rows[0]:
        return _plot_generic_csv(path, output)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    x = [_float(r, "macs") for r in rows]
    y = [_float(r, "macro_f1") for r in rows]
    _draw_pareto_front(ax, x, y)
    for row, xx, yy in zip(rows, x, y):
        label = row.get("candidate_hash", "")[:8]
        if label:
            ax.annotate(label, (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("MACs (log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"Pareto front: {path.parent.parent.name}")
    _style_axes(ax)
    return _save(fig, output)


def _plot_generation_csv(path: Path, output: Path):
    return _plot_evaluations(path, output)


def _plot_history(path: Path, output: Path):
    import matplotlib.pyplot as plt

    rows = _read_csv(path)
    if not rows or "epoch" not in rows[0]:
        return _plot_generic_csv(path, output)
    columns = [
        c for c in rows[0]
        if c not in {"epoch", "fold"} and all(_float(row, c) is not None for row in rows)
    ]
    preferred = [c for c in columns if c in {"train_loss", "loss", "macro_f1", "accuracy", "val_macro_f1", "val_accuracy"}]
    columns = preferred or columns[:4]
    fig, axes = plt.subplots(len(columns), 1, figsize=(9.5, max(3.8, 2.5 * len(columns))), squeeze=False)
    for ax, column in zip(axes[:, 0], columns):
        ax.plot([_float(r, "epoch") for r in rows], [_float(r, column) for r in rows], marker="o", markersize=2.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(column)
        _style_axes(ax)
    fig.suptitle(f"Training history: {path.parent.name}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, output)


def _plot_refit_loss_directory(path: Path, output: Path):
    """Plot final refit training loss for every role/seed under a ``final`` directory."""
    import matplotlib.pyplot as plt

    histories = []
    for history_path in sorted(path.glob("*/[0-9]*/history.csv")):
        rows = _read_csv(history_path)
        values = [(_float(row, "epoch"), _float(row, "train_loss")) for row in rows]
        values = [(epoch, loss) for epoch, loss in values if epoch is not None and loss is not None]
        if values:
            role, seed = history_path.parent.parent.name, history_path.parent.name
            histories.append((role, seed, values))
    if not histories:
        raise ValueError(f"No final refit history.csv files found under {path}")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    role_colors = {
        "high_f1": "#1F77B4",
        "high_accuracy": "#1F77B4",
        "knee": "#2CA02C",
        "low_macs": "#FF7F0E",
        "low_cost": "#FF7F0E",
    }
    for role, seed, values in histories:
        epochs = [item[0] for item in values]
        losses = [item[1] for item in values]
        color = role_colors.get(role, "#9467BD")
        ax.plot(
            epochs,
            losses,
            color=color,
            alpha=0.48,
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label=f"{role} / seed {seed}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title(f"Final refit training loss: {path.parent.name}")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style_axes(ax)
    fig.tight_layout()
    return _save(fig, output)


def _plot_summary(path: Path, output: Path):
    import matplotlib.pyplot as plt

    rows = _read_csv(path)
    roles = [role for role in ROLE_ORDER if any(r.get("role") == role for r in rows)]
    if not roles:
        return _plot_generic_csv(path, output)
    grouped = {}
    for role in roles:
        selected = [r for r in rows if r.get("role") == role]
        grouped[role] = {
            key: float(np.mean([_float(r, key) for r in selected if _float(r, key) is not None]))
            for key in ("accuracy", "macro_f1", "parameters", "macs")
            if any(_float(r, key) is not None for r in selected)
        }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    positions = np.arange(len(roles))
    width = 0.36
    axes[0].bar(positions - width / 2, [grouped[r].get("accuracy", 0) for r in roles], width, label="Accuracy", color="#4C78A8")
    axes[0].bar(positions + width / 2, [grouped[r].get("macro_f1", 0) for r in roles], width, label="Macro-F1", color="#F58518")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Final test metrics")
    axes[0].legend(frameon=False)
    axes[1].bar(positions - width / 2, [grouped[r].get("parameters", 0) / 1e6 for r in roles], width, label="Parameters", color="#54A24B")
    axes[1].bar(positions + width / 2, [grouped[r].get("macs", 0) / 1e6 for r in roles], width, label="MACs", color="#E45756")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Millions (log scale)")
    axes[1].set_title("Model size and compute")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(positions, roles, rotation=25, ha="right")
        _style_axes(ax)
    fig.suptitle(f"Final summary: {path.parent.parent.name}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, output)


def _plot_confusion_csv(path: Path, output: Path):
    import matplotlib.pyplot as plt

    rows = _read_csv(path)
    if not rows:
        return _plot_generic_csv(path, output)
    labels = [r.get("true_label", str(i)) for i, r in enumerate(rows)]
    columns = [c for c in rows[0] if c != "true_label"]
    matrix = np.asarray([[float(row[c]) for c in columns] for row in rows])
    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 0.65), max(5, len(labels) * 0.55)))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix: {path.parent.name}")
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    return _save(fig, output)


def _plot_generic_csv(path: Path, output: Path):
    import matplotlib.pyplot as plt

    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    columns = list(rows[0])
    numeric = [c for c in columns if all(_float(r, c) is not None for r in rows)]
    if numeric:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(rows))
        for column in numeric[:8]:
            ax.plot(x, [_float(r, column) for r in rows], marker=".", linewidth=1, label=column)
        ax.set_xlabel("Row index")
        ax.set_title(path.name)
        ax.legend(frameon=False, ncol=2)
        _style_axes(ax)
        return _save(fig, output)
    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.35 + 1)))
    ax.axis("off")
    table = ax.table(cellText=[[r.get(c, "") for c in columns] for r in rows[:40]], colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title(f"{path.name} (first {min(40, len(rows))} rows)", pad=14)
    return _save(fig, output)


def _plot_generation_json(path: Path, output: Path):
    import matplotlib.pyplot as plt

    files = sorted(path.parent.glob("generation_*.json"))
    records = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if "generation" in data and "hypervolume" in data:
                records.append(data)
        except json.JSONDecodeError:
            continue
    if not records:
        return _plot_json_table(path, output)
    records.sort(key=lambda r: r["generation"])
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot([r["generation"] for r in records], [r["hypervolume"] for r in records], marker="o", color="#4C78A8")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Search hypervolume: {path.parent.parent.parent.name}")
    _style_axes(ax)
    return _save(fig, output)


def _plot_metrics_json(path: Path, output: Path):
    import matplotlib.pyplot as plt

    data = json.loads(path.read_text(encoding="utf-8"))
    fold_metrics = data.get("fold_metrics")
    if isinstance(fold_metrics, list) and fold_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        folds = [item.get("fold", i) + 1 for i, item in enumerate(fold_metrics)]
        axes[0].bar(folds, [item.get("macro_f1", 0) for item in fold_metrics], color="#4C78A8")
        axes[0].set_title("Macro-F1 by fold")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylim(0, 1.05)
        axes[1].bar(folds, [item.get("accuracy", 0) for item in fold_metrics], color="#F58518")
        axes[1].set_title("Accuracy by fold")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            _style_axes(ax)
        fig.suptitle(f"Fold metrics: {path.parent.name}", fontsize=15, fontweight="bold")
        fig.tight_layout()
        return _save(fig, output)
    return _plot_json_table(path, output)


def _plot_representatives_json(path: Path, output: Path):
    import matplotlib.pyplot as plt

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [(role, value) for role, value in data.items() if isinstance(value, dict) and _float(value, "macs") is not None]
    if not rows:
        return _plot_json_table(path, output)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for role, value in rows:
        x, y = _float(value, "macs"), _float(value, "macro_f1")
        ax.scatter([x], [y], s=100, label=role)
        ax.annotate(role, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("MACs (log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Selected representative architectures")
    ax.legend(frameon=False)
    _style_axes(ax)
    return _save(fig, output)


def _plot_json_table(path: Path, output: Path):
    import matplotlib.pyplot as plt

    data = json.loads(path.read_text(encoding="utf-8"))
    flat = []

    def visit(prefix: str, value: Any):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else key, child)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat.append((prefix, str(value)))

    visit("", data)
    fig, ax = plt.subplots(figsize=(12, max(4, len(flat) * 0.28 + 1)))
    ax.axis("off")
    table = ax.table(cellText=flat, colLabels=["Field", "Value"], loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    ax.set_title(path.name, pad=14)
    return _save(fig, output)


def _plot_npz(path: Path, output: Path):
    import matplotlib.pyplot as plt

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"targets", "predictions"}.issubset(keys):
            targets = data["targets"].astype(int).reshape(-1)
            predictions = data["predictions"].astype(int).reshape(-1)
            labels = sorted(set(targets.tolist()) | set(predictions.tolist()))
            matrix = np.zeros((len(labels), len(labels)), dtype=int)
            lookup = {label: i for i, label in enumerate(labels)}
            for target, prediction in zip(targets, predictions):
                matrix[lookup[int(target)], lookup[int(prediction)]] += 1
            fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.65), max(4.5, len(labels) * 0.55)))
            image = ax.imshow(matrix, cmap="Blues")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(labels)), labels)
            ax.set_yticks(np.arange(len(labels)), labels)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(f"Predictions: {path.name}")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)
            fig.tight_layout()
            return _save(fig, output)
        if {"mean", "std"}.issubset(keys):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            mean, std = data["mean"].reshape(-1), data["std"].reshape(-1)
            x = np.arange(len(mean))
            ax.plot(x, mean, label="mean", color="#4C78A8")
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="#4C78A8", label="± std")
            ax.set_xlabel("Channel")
            ax.set_title(f"Normalization statistics: {path.name}")
            ax.legend(frameon=False)
            _style_axes(ax)
            return _save(fig, output)
        split_keys = sorted(k for k in keys if "train" in k or "validation" in k or "inner" in k)
        if split_keys:
            labels, sizes = [], []
            for key in split_keys:
                labels.append(key)
                sizes.append(len(data[key]))
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.bar(np.arange(len(labels)), sizes, color="#54A24B")
            ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"Saved split sizes: {path.name}")
            _style_axes(ax)
            return _save(fig, output)
    raise ValueError(f"No supported visual arrays found in {path}")


def _plot_log(path: Path, output: Path):
    import matplotlib.pyplot as plt

    loss, f1, accuracy, train_loss = [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"epoch=(\d+)/(\d+).*?loss=([0-9.eE+-]+).*?val_macro_f1=([0-9.eE+-]+).*?val_accuracy=([0-9.eE+-]+)", line)
        if match:
            loss.append(float(match.group(3))); f1.append(float(match.group(4))); accuracy.append(float(match.group(5)))
            continue
        match = re.search(r"epoch=(\d+)/(\d+).*?train_loss=([0-9.eE+-]+)", line)
        if match:
            train_loss.append(float(match.group(3)))
    if not loss and not train_loss:
        raise ValueError(f"No epoch metrics recognized in {path}")
    panels = 2 if loss else 1
    fig, axes = plt.subplots(panels, 1, figsize=(10, 6 if panels == 2 else 4), squeeze=False)
    if loss:
        ax = axes[0, 0]
        ax.plot(loss, label="validation loss")
        ax.plot(f1, label="validation Macro-F1")
        ax.plot(accuracy, label="validation Accuracy")
        ax.set_title("Search epoch metrics")
        ax.legend(frameon=False)
        _style_axes(ax)
        axes[1, 0].plot(train_loss, color="#54A24B")
        axes[1, 0].set_title("Refit train loss")
        _style_axes(axes[1, 0])
    else:
        axes[0, 0].plot(train_loss, color="#54A24B")
        axes[0, 0].set_title("Training loss")
        _style_axes(axes[0, 0])
    fig.suptitle(path.name, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, output)


def _plot_dataset_directory(path: Path, output: Path):
    import matplotlib.pyplot as plt

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    evaluations = path / "search" / "evaluations.csv"
    if evaluations.exists():
        rows = [r for r in _read_csv(evaluations) if r.get("status", "completed") == "completed"]
        if rows:
            axes[0].scatter(
                [_float(r, "macs") for r in rows],
                [_float(r, "macro_f1") for r in rows],
                s=18,
                alpha=0.35,
                color="#9E9E9E",
                edgecolors="none",
                label="all candidates",
            )
            pareto_path = path / "search" / "pareto_front.csv"
            if pareto_path.exists():
                pareto = _read_csv(pareto_path)
                px = np.asarray([_float(r, "macs") for r in pareto], dtype=float)
                py = np.asarray([_float(r, "macro_f1") for r in pareto], dtype=float)
                _draw_pareto_front(axes[0], px, py)
            axes[0].set_xscale("log")
            axes[0].set_xlabel("MACs (log)")
            axes[0].set_ylabel("Macro-F1")
            axes[0].set_title("All candidates")
            axes[0].legend(frameon=False, fontsize=8)
    hypervolume_files = sorted((path / "search" / "generations").glob("generation_*.json"))
    hv = []
    for file in hypervolume_files:
        try:
            item = json.loads(file.read_text(encoding="utf-8"))
            if "generation" in item and "hypervolume" in item:
                hv.append((item["generation"], item["hypervolume"]))
        except json.JSONDecodeError:
            pass
    if hv:
        axes[1].plot([x[0] for x in hv], [x[1] for x in hv], marker="o")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Hypervolume")
        axes[1].set_title("Search convergence")
    summary_path = path / "final" / "summary.csv"
    if summary_path.exists():
        rows = _read_csv(summary_path)
        roles = [r for r in ROLE_ORDER if any(x.get("role") == r for x in rows)]
        means = {role: np.mean([_float(x, "macro_f1") for x in rows if x.get("role") == role]) for role in roles}
        axes[2].bar(roles, [means[r] for r in roles], color="#F58518")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_ylabel("Macro-F1")
        axes[2].set_title("Final Macro-F1 by role")
        axes[2].tick_params(axis="x", rotation=25)
    reps_path = path / "search" / "representatives.json"
    if reps_path.exists():
        reps = json.loads(reps_path.read_text(encoding="utf-8"))
        styles = {
            "high_f1": ("#1F77B4", "*"),
            "high_accuracy": ("#1F77B4", "*"),
            "knee": ("#2CA02C", "D"),
            "low_macs": ("#FF7F0E", "P"),
            "low_cost": ("#FF7F0E", "P"),
        }
        for role, row in reps.items():
            if isinstance(row, dict) and _float(row, "macs") is not None:
                color, marker = styles.get(role, ("#9467BD", "o"))
                axes[3].scatter([_float(row, "macs")], [_float(row, "macro_f1")], s=120, color=color, marker=marker, edgecolors="black", linewidths=0.8, label=role)
        axes[3].set_xscale("log")
        axes[3].set_xlabel("MACs (log)")
        axes[3].set_ylabel("Macro-F1")
        axes[3].set_title("Selected representatives")
        axes[3].legend(frameon=False)
    for ax in axes:
        _style_axes(ax)
    dataset = manifest.get("dataset", path.name)
    fig.suptitle(f"Mamba-NAS run overview: {dataset}", fontsize=17, fontweight="bold")
    fig.text(0.01, 0.01, f"run={manifest.get('run_id', path.parent.name)} | strategy={manifest.get('split', {}).get('strategy', 'unknown')}", fontsize=9, color="#59636E")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return _save(fig, output)


def plot_artifact(input_path: str | Path, output: str | Path | None = None) -> Path:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    destination = _output_path(path, output)
    if path.is_dir():
        if path.name == "final":
            return _plot_refit_loss_directory(path, destination)
        return _plot_dataset_directory(path, destination)
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".csv":
        if name == "evaluations.csv":
            return _plot_evaluations(path, destination)
        if name == "pareto_front.csv" or name.endswith("_pareto.csv"):
            evaluations_path = path.with_name("evaluations.csv")
            if evaluations_path.exists():
                return _plot_evaluations(evaluations_path, destination)
            return _plot_pareto(path, destination)
        if name == "summary.csv":
            return _plot_summary(path, destination)
        if name == "confusion_matrix.csv":
            return _plot_confusion_csv(path, destination)
        if name == "history.csv" or "history" in name:
            return _plot_history(path, destination)
        if re.match(r"generation_\d+\.csv", name):
            return _plot_generation_csv(path, destination)
        return _plot_generic_csv(path, destination)
    if suffix == ".json":
        if re.match(r"generation_\d+\.json", name):
            return _plot_generation_json(path, destination)
        if name == "metrics.json":
            return _plot_metrics_json(path, destination)
        if name == "representatives.json":
            return _plot_representatives_json(path, destination)
        return _plot_json_table(path, destination)
    if suffix == ".npz":
        return _plot_npz(path, destination)
    if suffix in {".log", ".txt"}:
        return _plot_log(path, destination)
    raise ValueError(f"Unsupported artifact type: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="A file or dataset directory under runs/")
    parser.add_argument("--output", help="PNG destination; defaults to <input>.png or <input>/run_overview.png")
    args = parser.parse_args()
    print(plot_artifact(args.input, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
