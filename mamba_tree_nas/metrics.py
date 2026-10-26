from __future__ import annotations

import math
import numpy as np


def classification(labels, probabilities, classes, ce=None):
    labels = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p.argmax(1)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, pred), 1)
    tp = np.diag(confusion).astype(float)
    support, predicted = confusion.sum(1), confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(classes), where=precision + recall > 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / len(labels)), "balanced_accuracy": float(recall.mean()), "ce": float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-300, 1)).mean()) if ce is None else float(ce), "per_class": [{"class": i, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i in range(classes)], "confusion_matrix": confusion.tolist()}


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fronts(objectives):
    remaining = list(range(len(objectives)))
    result = []
    while remaining:
        front = [i for i in remaining if not any(dominates(objectives[j], objectives[i]) for j in remaining if i != j)]
        result.append(front)
        chosen = set(front)
        remaining = [i for i in remaining if i not in chosen]
    return result


def crowding(objectives, front):
    values = {i: 0.0 for i in front}
    for dimension in range(2):
        ordered = sorted(front, key=lambda i: (objectives[i][dimension], i))
        if not ordered:
            continue
        values[ordered[0]] = values[ordered[-1]] = float("inf")
        span = objectives[ordered[-1]][dimension] - objectives[ordered[0]][dimension]
        if span > 0:
            for j in range(1, len(ordered) - 1):
                values[ordered[j]] += (objectives[ordered[j + 1]][dimension] - objectives[ordered[j - 1]][dimension]) / span
    return values


def nsga_select(records, count):
    objectives = [r["objectives"] for r in records]
    selected = []
    for front in fronts(objectives):
        distances = crowding(objectives, front)
        ordered = sorted(front, key=lambda i: (-distances[i], records[i]["hash"]))
        selected.extend(ordered[:count - len(selected)])
        if len(selected) == count:
            break
    return [records[i] for i in selected]


def tournament(records, rng):
    objectives = [r["objectives"] for r in records]
    ranks, distances = {}, {}
    for rank, front in enumerate(fronts(objectives)):
        ranks.update({i: rank for i in front}); distances.update(crowding(objectives, front))
    a, b = rng.choices(range(len(records)), k=2)
    i = min((a, b), key=lambda i: (ranks[i], -distances[i], records[i]["hash"]))
    return records[i]


def pareto(records):
    successful = [r for r in records if r.get("status") == "complete"]
    return [successful[i] for i in fronts([r["objectives"] for r in successful])[0]] if successful else []


def normalized_objective(record, cost_bound):
    return [record["objectives"][0], record["objectives"][1] / math.log1p(cost_bound)]


def hypervolume(points, reference=(1.1, 1.1)):
    # Exact 2-D minimization union of rectangles; duplicate points are harmless.
    ordered = sorted((float(x), float(y)) for x, y in points if x < reference[0] and y < reference[1])
    total, last_y = 0.0, reference[1]
    for x, y in ordered:
        if y < last_y:
            total += (reference[0] - x) * (last_y - y)
            last_y = y
    return total


def representatives(records, cost_bound):
    front = pareto(records)
    if not front:
        return {}
    tie = lambda r: (r["macs"], r["hash"])
    return {"high_f1": min(front, key=lambda r: (r["objectives"][0], *tie(r)))["hash"], "low_macs": min(front, key=tie)["hash"], "knee": min(front, key=lambda r: (sum(x*x for x in normalized_objective(r, cost_bound)), *tie(r)))["hash"]}


def topology_stats(records):
    from collections import Counter
    from .architecture import Architecture
    counts = Counter(Architecture.from_dict(r["architecture"]).tree.shape for r in records if r.get("status") == "complete")
    total = sum(counts.values())
    entropy = -sum((v / total) * math.log(v / total) for v in counts.values()) if total else 0.0
    return {"topology_count": len(counts), "topology_entropy": entropy, "topology_counts": dict(counts)}


def proxy_diagnostics(records, top_fraction=0.1):
    from scipy.stats import spearmanr
    output = []
    fields = ("gradient_l2", "gradient_rms", "sum_abs_gtheta", "mean_abs_gtheta", "absolute_cosine", "product", "log10_mean_product")
    for scope in ("full", "trunk"):
        for field in fields:
            pairs = [(r["proxy"].get("aggregate", {}).get(scope, {}).get(field), r["validation"]["macro_f1"]) for r in records if r.get("status") == "complete" and r.get("proxy", {}).get("status") == "complete" and all(repeat.get(scope, {}).get("status") == "ok" for repeat in r["proxy"].get("repeats", []))]
            pairs = [(float(x), y) for x, y in pairs if x is not None and np.isfinite(x)]
            n = len(pairs)
            if n < 2:
                output.append({"scope": scope, "field": field, "n": n, "spearman": None, "top_fraction": top_fraction, "ascending_hit": None, "descending_hit": None}); continue
            x, y = np.array(pairs).T
            rho = float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None
            k = max(1, math.ceil(top_fraction * n))
            target = set(np.argsort(-y, kind="stable")[:k])
            output.append({"scope": scope, "field": field, "n": n, "spearman": rho, "top_fraction": top_fraction, "ascending_hit": len(target & set(np.argsort(x, kind="stable")[:k])) / k, "descending_hit": len(target & set(np.argsort(-x, kind="stable")[:k])) / k})
    return output


def nested_summary(rows):
    """Rows represent unique architecture/seed fits with role aliases expanded."""
    import statistics
    grouped = {}
    for r in rows:
        key = (r["dataset"], r["variant"], r["role"], r["search_seed"], r.get("run_id", ""))
        grouped.setdefault(key, []).append(r)
    within = []
    for key, values in sorted(grouped.items()):
        item = dict(zip(("dataset", "variant", "role", "search_seed", "run_id"), key))
        item["refit_count"] = len(values)
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "ce"):
            scores = [v[metric] for v in values]
            item[metric + "_mean"] = statistics.mean(scores)
            item[metric + "_within_std"] = statistics.stdev(scores) if len(scores) > 1 else None
        for metric in ("parameters", "macs", "batch1_median_ms", "batch1_p90_ms", "throughput_samples_per_second", "peak_allocated_gpu_bytes", "training_peak_allocated_gpu_bytes", "training_seconds"):
            scores = [v[metric] for v in values if v.get(metric) is not None]
            if scores:
                item[metric + "_mean"] = statistics.mean(scores)
                item[metric + "_within_std"] = statistics.stdev(scores) if len(scores) > 1 else None
        within.append(item)
    outer = {}
    for r in within:
        key = (r["dataset"], r["variant"], r["role"])
        outer.setdefault(key, []).append(r)
    summary = []
    for key, values in sorted(outer.items()):
        item = dict(zip(("dataset", "variant", "role"), key))
        item["search_count"] = len(values)
        item["independent_search_seeds"] = len({v["search_seed"] for v in values})
        item["protocol"] = "mean across searches of within-search refit means; std across searches"
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "ce"):
            scores = [v[metric + "_mean"] for v in values]
            item[metric + "_mean"] = statistics.mean(scores)
            item[metric + "_between_std"] = statistics.stdev(scores) if len(scores) > 1 else None
            stds = [v[metric + "_within_std"] for v in values if v[metric + "_within_std"] is not None]
            item[metric + "_mean_within_std"] = statistics.mean(stds) if stds else None
        for metric in ("parameters", "macs", "batch1_median_ms", "batch1_p90_ms", "throughput_samples_per_second", "peak_allocated_gpu_bytes", "training_peak_allocated_gpu_bytes", "training_seconds"):
            scores = [v[metric + "_mean"] for v in values if v.get(metric + "_mean") is not None]
            if scores:
                item[metric + "_mean"] = statistics.mean(scores)
                item[metric + "_between_std"] = statistics.stdev(scores) if len(scores) > 1 else None
        summary.append(item)
    return within, summary
