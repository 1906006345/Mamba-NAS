"""Dimension-aware NSGA-II and comparable deployment study metrics."""
import math
import statistics
from collections import Counter

from .metrics import fronts, hypervolume as hypervolume_2d


def hypervolume(points, reference=(1.1, 1.1)):
    ref = tuple(reference)
    points = [tuple(p) for p in points]
    if len(ref) not in (2, 3) or not all(map(math.isfinite, ref)):
        raise ValueError("HV requires a finite 2D or 3D reference")
    if any(len(p) != len(ref) or not all(map(math.isfinite, p)) for p in points):
        raise ValueError("Invalid HV point dimension or value")
    points = sorted({p for p in points if all(x < r for x, r in zip(p, ref))})
    if len(ref) == 2:
        return hypervolume_2d(points, ref)
    # Slice the exact union of axis-aligned boxes along the first axis.
    area, volume, previous = [], 0., None
    for point in points:
        if previous is not None and point[0] != previous:
            volume += (point[0] - previous) * hypervolume_2d(area, ref[1:])
        area.append(point[1:])
        previous = point[0]
    return volume + (ref[0] - previous) * hypervolume_2d(area, ref[1:]) if area else 0.


def nsga_order(records):
    """Pareto rank, then descending crowding distance, then stable hash."""
    if not records:
        return []
    points = [row["objectives"] for row in records]
    dimensions = len(points[0])
    if dimensions not in (2, 3) or any(len(p) != dimensions or not all(map(math.isfinite, p)) for p in points):
        raise ValueError("Invalid NSGA-II objectives")
    result = []
    for front in fronts(points):
        distance = {i: 0. for i in front}
        for dimension in range(dimensions):
            ordered = sorted(front, key=lambda i: (points[i][dimension], i))
            distance[ordered[0]] = distance[ordered[-1]] = math.inf
            span = points[ordered[-1]][dimension] - points[ordered[0]][dimension]
            if span:
                for j in range(1, len(ordered) - 1):
                    distance[ordered[j]] += (points[ordered[j + 1]][dimension]
                                              - points[ordered[j - 1]][dimension]) / span
        result.extend(sorted(front, key=lambda i: (-distance[i], records[i]["hash"])))
    return [records[i] for i in result]


def curves_at(rows, key, limit):
    """Integrate the right-continuous step curve from zero up to a common limit."""
    if limit <= 0:
        return 0.
    position, value, area = 0., 0., 0.
    for row in rows:
        point = min(row[key], limit)
        if point < position:
            raise ValueError("Nonmonotonic curve coordinates")
        area += (point - position) * value
        if row[key] > limit:
            return area / limit
        position, value = point, row["hv"]
    return (area + (limit - position) * value) / limit


def hv_at(rows, key, limit):
    return next((row["hv"] for row in reversed(rows) if row[key] <= limit), 0.)


def correlation(x, y):
    from scipy.stats import kendalltau, rankdata

    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"kendall_tau_b": None, "spearman": None}
    # Spearman is Pearson correlation of average ranks, including tied observations.
    ranks = [rankdata(values, method="average").tolist() for values in (x, y)]
    return {"kendall_tau_b": float(kendalltau(x, y, variant="b").statistic),
            "spearman": statistics.correlation(*ranks)}


def rank_summary(rows):
    paired = [r for r in rows if r["short_f1"] is not None and r["reference_f1"] is not None]
    fields = ("short_f1", "reference_f1")
    ranks = [sorted(paired, key=lambda r: (-r[field], r["hash"])) for field in fields]
    k = min(5, len(paired))
    summary = {"architectures": len(rows), "valid_pairs": len(paired), "failed_pairs": len(rows) - len(paired),
               "top_k": k, "top_k_overlap": None, "selection_regret_f1_pp": None}
    values = [[r[field] for r in paired] for field in fields]
    summary.update(correlation(*values))
    for field, scores in zip(fields, values):
        summary[field + "_tied_architectures"] = sum(n for n in Counter(scores).values() if n > 1)
    if k:
        summary["top_k_overlap"] = len({r["hash"] for r in ranks[0][:k]} & {r["hash"] for r in ranks[1][:k]}) / k
        summary["selection_regret_f1_pp"] = 100 * (ranks[1][0]["reference_f1"] - ranks[0][0]["reference_f1"])
    return summary


def cost_order_summary(rows):
    from itertools import combinations

    valid = [r for r in rows if r["status"] == "complete"]
    macs = [r["macs"] for r in valid]
    latency = [r["latency_ms"] for r in valid]
    comparable, reversed_pairs, ties = 0, 0, 0
    for a, b in combinations(valid, 2):
        cost_delta = a["macs"] - b["macs"]
        latency_delta = a["latency_ms"] - b["latency_ms"]
        if cost_delta:
            comparable += 1
            reversed_pairs += cost_delta * latency_delta < 0
            ties += latency_delta == 0
    return dict(correlation(macs, latency), architectures=len(rows), valid=len(valid),
                distinct_macs_pairs=comparable, inversion_pairs=reversed_pairs, latency_ties=ties,
                inversion_fraction=reversed_pairs / comparable if comparable else None)
