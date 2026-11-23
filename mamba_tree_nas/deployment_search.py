"""Paired NSGA-II runs, frozen hardware objectives and independent front audits."""
import math
import random
import time

from .architecture import Architecture
from .artifacts import Events, error_record, read_json, write_csv
from .cost import macs
from .deployment_benchmark import profile
from .deployment_metrics import hypervolume, nsga_order
from .deployment_rank import score_seeds, summarize_seeds
from .followup_evaluation import atomic_json
from .metrics import fronts
from .search import tuple_tree

ARMS = ("f1_macs", "f1_latency", "f1_latency_memory")
SUPERNET = "supernet_latency_memory"


def _pending(study, rng, records, population, count):
    space, config = study.space, study.cfg["search"]
    used = {r["hash"] for r in records}
    values = []
    ranked = {r["hash"]: i for i, r in enumerate(nsga_order(population))}

    def parent():
        selected = min(rng.choices(population, k=2), key=lambda r: ranked[r["hash"]])
        return Architecture.from_dict(selected["architecture"])

    for _ in range(space.c["max_attempts"]):
        if not population:
            child = space.sample(rng)
        else:
            a, b = parent(), parent()
            child = space.crossover(a, b, rng) if rng.random() < config["crossover_probability"] else a
            if rng.random() < config["mutation_probability"]:
                child = space.mutate(child, rng, config["mutation_weights"])
        if child.hash not in used:
            used.add(child.hash)
            values.append(child.to_dict())
            if len(values) == count:
                return values
    raise RuntimeError("Unique candidate generation exhausted the standard search operators")


def _common(f1, hardware, scales):
    return [1 - f1, hardware["latency_ms"] / scales["latency_ms"], hardware["peak_bytes"] / scales["peak_bytes"]]


def _score(study, arch, seed, bank, pretraining):
    if bank is None:
        result, hit = study.standalone(arch, study.cfg["seeds"]["candidate"], "short")
        return result, hit

    def inherited():
        from .training import evaluate

        model = bank.export(arch).to(study.cfg["device"])
        metrics, _ = evaluate(model, study.data.val, study.cfg, len(study.data.classes))
        counts = bank.activation_counts(arch, pretraining["activation_counts"])
        return {"validation": metrics, "weight_group_activations": counts,
                "untrained_weight_groups": [k for k, v in counts.items() if not v]}

    return study.cached("inherited", {"hash": arch.hash, "supernet_seed": seed}, inherited)


def _evaluate(study, arch, arm, seed, bank, pretraining, scales, row):
    start = time.perf_counter()
    hardware, hits = profile(study, arch)
    row.update(hardware=hardware, profiling_cache_hits=hits, profiling_wall_seconds=time.perf_counter() - start)
    start = time.perf_counter()
    result, hit = _score(study, arch, seed, bank, pretraining)
    row.update(score=result, score_cache_hit=hit, scoring_wall_seconds=time.perf_counter() - start)
    f1 = result["validation"]["macro_f1"]
    if not math.isfinite(f1) or not 0 <= f1 <= 1:
        raise FloatingPointError("Invalid validation F1")
    common = _common(f1, hardware, scales)
    objectives = ([common[0], math.log1p(row["macs"])] if arm == "f1_macs" else
                  common[:2] if arm == "f1_latency" else common)
    row.update(status="complete", validation_f1=f1, common=common, objectives=objectives)


def _history(records, ref, start_wall=0., start_attributed=0.):
    wall, attributed = start_wall, start_attributed
    points, native, rows = [], [], []
    native_dimension = next((len(r["objectives"]) for r in records if r["status"] == "complete"), len(ref))
    for row in records:
        wall = row["wall_seconds"]
        attributed += row.get("hardware", {}).get("elapsed_seconds", 0.) + row.get("score", {}).get("elapsed_seconds", 0.)
        if row["status"] == "complete":
            points.append(row["common"])
            native.append(row["native_hv_coordinates"])
        rows.append({"candidates": row["evaluation_index"], "wall_seconds": wall,
                     "attributed_seconds": attributed, "hv": hypervolume(points, ref),
                     "native_hv": hypervolume(native, ref[:native_dimension]),
                     "projected_hv": hypervolume([p[:2] for p in points], ref[:2]),
                     "successful": len(points),
                     "outside_reference": sum(any(x >= lim for x, lim in zip(p, ref)) for p in points)})
    return rows


def search_path(study, arm, seed):
    return study.root / "search" / arm / str(seed)


def run_search(study, arm, seed, interrupt_after=None):
    from .cost import cost_upper_bound

    cfg, settings = study.cfg, study.settings
    config = cfg["search"]
    scales_path = study.root / "benchmark" / "scales.json"
    if not scales_path.exists():
        raise ValueError(f"Run benchmark --execute before searching: {study.data.name}")
    scales = read_json(scales_path)
    ref = settings["hv_reference"]
    output = search_path(study, arm, seed)
    path = output / "state.json"
    rng = random.Random(seed)
    records = []
    bank, pretraining = None, None
    if path.exists():
        state = read_json(path)
        if state["protocol"] != study.identity or state["scales"] != scales:
            raise ValueError("Deployment search protocol or frozen scales changed")
        records = state["records"]
        rng.setstate(tuple_tree(state["rng"]))
        if state["status"] == "complete":
            _export_search(study, output, state)
            return state
    else:
        if arm == SUPERNET:
            from .deployment_supernet import train_supernet

            bank, pretraining = train_supernet(study, seed)
        start_seconds = pretraining["elapsed_seconds"] if pretraining else 0.
        state = {"status": "running", "protocol": study.identity, "scales": scales, "records": records,
                 "generation": 0, "pending": [], "population": [], "rng": rng.getstate(), "arm": arm,
                 "seed": seed, "pretraining_seconds": start_seconds, "wall_seconds": start_seconds}
        atomic_json(path, state)
    if arm == SUPERNET and bank is None:
        from .deployment_supernet import train_supernet

        bank, pretraining = train_supernet(study, seed)
    cost_bound = cost_upper_bound(cfg["space"], *study.dimensions)
    indexed = {r["hash"]: r for r in records}
    origin = time.perf_counter()
    offset = state["wall_seconds"]
    events = Events(output / "tensorboard", cfg, len(records) + 1 if records else None)
    consecutive_failures = 0

    def save():
        state.update(wall_seconds=offset + time.perf_counter() - origin, rng=rng.getstate())
        atomic_json(path, state)

    try:
        state["status"] = "running"
        while len(records) < config["budget"]:
            if not state["pending"]:
                count = min(config["population"], config["budget"] - len(records))
                state["pending"] = _pending(study, rng, records, [indexed[h] for h in state["population"]], count)
                save()
            arch = Architecture.from_dict(state["pending"][0])
            row = {"hash": arch.hash, "architecture": arch.to_dict(), "generation": state["generation"],
                   "evaluation_index": len(records) + 1, "macs": macs(arch, *study.dimensions)}
            try:
                _evaluate(study, arch, arm, seed, bank, pretraining, scales, row)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                row.update(status="failed", error=error_record(exc, "candidate"))
                consecutive_failures += 1
            else:
                row["native_hv_coordinates"] = ([row["common"][0], math.log1p(row["macs"]) / math.log1p(cost_bound)]
                                                if arm == "f1_macs" else row["objectives"])
                consecutive_failures = 0
            row["wall_seconds"] = offset + time.perf_counter() - origin
            records.append(row)
            indexed[arch.hash] = row
            state["pending"].pop(0)
            if not state["pending"]:
                pool = [indexed[h] for h in state["population"]] + [
                    r for r in records if r["generation"] == state["generation"] and r["status"] == "complete"]
                state["population"] = [r["hash"] for r in nsga_order(pool)[:config["population"]]]
                state["generation"] += 1
            save()
            points = [r["common"] for r in records if r["status"] == "complete"]
            events.scalar("search/common_hv", hypervolume(points, ref), len(records))
            events.scalar("search/elapsed_seconds", state["wall_seconds"], len(records))
            print(f"[SEARCH] {study.data.name} {arm} seed={seed} {len(records)}/{config['budget']} {row['status']}", flush=True)
            if consecutive_failures == config["consecutive_failure_limit"]:
                raise RuntimeError("Too many consecutive candidate failures; inspect state.json and cache error records")
            if interrupt_after == len(records):
                raise KeyboardInterrupt("Deployment search test interruption")
        state["status"] = "complete"
        save()
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["last_error"] = error_record(exc, "search")
        save()
        raise
    finally:
        events.close()
    _export_search(study, output, state)
    return state


def _export_search(study, output, state):
    ref = study.settings["hv_reference"]
    records = state["records"]
    start = state["pretraining_seconds"]
    write_csv(output / "curve.csv", _history(records, ref, start, start))
    valid = [r for r in records if r["status"] == "complete"]
    layers = fronts([r["common"] for r in valid]) if valid else []
    write_csv(output / "pareto.csv", [valid[i] for i in layers[0]] if layers else [])
    projection = fronts([r["common"][:2] for r in valid]) if valid else []
    write_csv(output / "f1_latency_front.csv", [
        {"hash": valid[i]["hash"], "validation_f1": valid[i]["validation_f1"],
         "latency_ms": valid[i]["hardware"]["latency_ms"], "peak_bytes": valid[i]["hardware"]["peak_bytes"],
         "macs": valid[i]["macs"]}
        for i in projection[0]
    ] if projection else [])


def audit_front(study, arm, seed, state):
    output = search_path(study, arm, seed)
    path = output / "audit.json"
    if path.exists():
        previous = read_json(path)
        if previous["curve"][-1]["failed_audits"] == 0:
            return previous["curve"]
    started = time.perf_counter()
    history = {r["candidates"]: r for r in _history(state["records"], study.settings["hv_reference"],
                                                   state["pretraining_seconds"], state["pretraining_seconds"])}
    selected, details, curve = {}, [], []
    for count in study.settings["checkpoints"]:
        candidates = [r for r in state["records"][:count] if r["status"] == "complete" and r["hash"] not in selected]
        quota = study.settings["audit_quota"]
        for row in nsga_order(candidates)[:quota]:
            arch = Architecture.from_dict(row["architecture"])
            results = score_seeds(study, arch, "reference")
            f1, sd = summarize_seeds(results)
            common = _common(f1, row["hardware"], state["scales"]) if f1 is not None else None
            selected[row["hash"]] = common
            details.append({"hash": row["hash"], "checkpoint_candidates": count, "reference_f1": f1,
                            "reference_seed_sd": sd, "common": common, "fits": results})
        points = [p for p in selected.values() if p is not None]
        ref = study.settings["hv_reference"]
        curve.append({**history[count], "online_hv": history[count]["hv"],
                      "hv": hypervolume(points, ref), "projected_hv": hypervolume([p[:2] for p in points], ref[:2]),
                      "audited": len(selected), "successful_audits": len(points),
                      "failed_audits": len(selected) - len(points),
                      "audit_attributed_seconds": sum(f.get("elapsed_seconds", 0.) for r in details for f in r["fits"]),
                      "audit_new_fit_seconds": sum(f.get("elapsed_seconds", 0.) for r in details for f in r["fits"]
                                                   if not f.get("cached"))})
        print(f"[AUDIT] {study.data.name} {arm} seed={seed} candidates={count} "
              f"audited={len(points)}/{len(selected)}", flush=True)
    write_csv(output / "audit_candidates.csv", details)
    write_csv(output / "audit_curve.csv", curve)
    atomic_json(path, {"curve": curve, "candidates": details, "elapsed_seconds": time.perf_counter() - started})
    return curve
