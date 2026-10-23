"""Reference-front and convergence metrics for two-objective NAS runs.

All functions operate on minimisation objectives.  Mamba-NAS stores
``objective_f1 = 1 - Macro-F1`` and ``objective_macs = log1p(MACs)``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def objective_array(rows: Iterable[dict]) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append([float(row["objective_f1"]), float(row["objective_macs"])])
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(values, dtype=float).reshape((-1, 2))


def nondominated_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape((-1, 2))
    if len(points) == 0:
        return points
    keep = []
    for index, point in enumerate(points):
        dominated = np.any(
            np.all(points <= point, axis=1)
            & np.any(points < point, axis=1)
            & (np.arange(len(points)) != index)
        )
        if not dominated:
            keep.append(point)
    return np.unique(np.asarray(keep), axis=0)


def reference_front(rows_or_points: Iterable[dict] | np.ndarray) -> np.ndarray:
    points = (
        np.asarray(rows_or_points, dtype=float)
        if isinstance(rows_or_points, np.ndarray)
        else objective_array(rows_or_points)
    )
    return nondominated_points(points)


def normalize_objectives(points: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points/reference normalized to [0, 1] using reference bounds."""
    points = np.asarray(points, dtype=float).reshape((-1, 2))
    reference = np.asarray(reference, dtype=float).reshape((-1, 2))
    if len(reference) == 0:
        return points, reference, np.ones(2)
    low = np.minimum(reference.min(axis=0), points.min(axis=0) if len(points) else reference.min(axis=0))
    high = np.maximum(reference.max(axis=0), points.max(axis=0) if len(points) else reference.max(axis=0))
    span = np.where(high > low, high - low, 1.0)
    return (points - low) / span, (reference - low) / span, span


def hypervolume(points: np.ndarray, reference: np.ndarray | None = None) -> float:
    points = nondominated_points(points)
    if len(points) == 0:
        return 0.0
    if reference is None:
        reference = np.asarray([1.1, max(1.0, float(points[:, 1].max()) * 1.01)])
    reference = np.asarray(reference, dtype=float)
    valid = points[np.all(points < reference, axis=1)]
    if len(valid) == 0:
        return 0.0
    ordered = valid[np.argsort(valid[:, 0])]
    area = 0.0
    previous_y = reference[1]
    for x, y in ordered:
        if y < previous_y:
            area += max(0.0, reference[0] - x) * (previous_y - y)
            previous_y = y
    return float(area)


def hv_auc(xs: Iterable[float], ys: Iterable[float], budget: float | None = None) -> float:
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float(y[valid][0] if valid.sum() else 0.0)
    order = np.argsort(x[valid], kind="mergesort")
    x, y = x[valid][order], y[valid][order]
    value = float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))
    denominator = float(budget if budget is not None and budget > 0 else max(x[-1], 1.0))
    return value / denominator


def _distances(current: np.ndarray, reference: np.ndarray, plus: bool = False) -> np.ndarray:
    if len(current) == 0 or len(reference) == 0:
        return np.asarray([], dtype=float)
    distances = []
    for ref in reference:
        delta = current - ref
        if plus:
            delta = np.maximum(delta, 0.0)
        distances.append(np.sqrt(np.sum(delta * delta, axis=1)).min())
    return np.asarray(distances, dtype=float)


def gd(current: np.ndarray, reference: np.ndarray) -> float:
    if len(current) == 0 or len(reference) == 0:
        return float("inf")
    return float(np.mean(_distances(reference, current, plus=False)))


def igd(current: np.ndarray, reference: np.ndarray) -> float:
    if len(current) == 0 or len(reference) == 0:
        return float("inf")
    return float(np.mean(_distances(current, reference, plus=False)))


def igd_plus(current: np.ndarray, reference: np.ndarray) -> float:
    if len(current) == 0 or len(reference) == 0:
        return float("inf")
    return float(np.mean(_distances(current, reference, plus=True)))


def epsilon_indicator(current: np.ndarray, reference: np.ndarray) -> float:
    """Additive epsilon indicator for minimisation objectives."""
    if len(current) == 0 or len(reference) == 0:
        return float("inf")
    return float(max(np.min(np.max(current - ref, axis=1)) for ref in reference))


def spacing(points: np.ndarray) -> float:
    points = nondominated_points(points)
    if len(points) < 3:
        return 0.0
    distances = []
    for index, point in enumerate(points):
        other = np.delete(points, index, axis=0)
        distances.append(np.min(np.abs(other - point).sum(axis=1)))
    return float(np.std(distances))


def spread(points: np.ndarray) -> float:
    points = nondominated_points(points)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def front_metrics(current: np.ndarray, reference: np.ndarray | None = None, hv_reference: np.ndarray | None = None) -> dict:
    current = nondominated_points(current)
    if reference is None:
        reference = current
    normalized_current, normalized_reference, _ = normalize_objectives(current, reference)
    # Metrics are measured after common normalization so MACs does not dominate.
    normalized_current = nondominated_points(normalized_current)
    normalized_reference = nondominated_points(normalized_reference)
    return {
        "hypervolume": hypervolume(current, hv_reference),
        "gd": gd(normalized_current, normalized_reference),
        "igd": igd(normalized_current, normalized_reference),
        "igd_plus": igd_plus(normalized_current, normalized_reference),
        "epsilon_indicator": epsilon_indicator(normalized_current, normalized_reference),
        "spacing": spacing(normalized_current),
        "spread": spread(normalized_current),
        "pareto_size": int(len(current)),
    }


def target_crossing(xs: Iterable[float], ys: Iterable[float], target: float, maximize: bool = True) -> float | None:
    pairs = sorted((float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y))
    for x, y in pairs:
        if (y >= target) if maximize else (y <= target):
            return x
    return None
