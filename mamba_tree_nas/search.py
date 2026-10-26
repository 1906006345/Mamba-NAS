from __future__ import annotations

import math
import random
import time
from pathlib import Path
import torch
from .architecture import Architecture, Space
from .artifacts import Events, atomic_json, error_record, implementation_fingerprint, init_run, read_json, run_lock, write_csv
from .config import digest, protocol_fingerprint, resolve_path
from .cost import COST_CONVENTION, cost_upper_bound, macs
from .data import DataProtocol
from .metrics import hypervolume, normalized_objective, nsga_select, pareto, representatives, topology_stats, tournament
from .proxy import measure_proxy
from .training import train


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, list) else value


def run_path(cfg, dataset, variant, seed, run_id):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
        raise ValueError("run-id must be a simple directory name")
    return resolve_path(cfg["output"]) / dataset / variant / f"seed_{seed}" / run_id


def load_records(path):
    state = read_json(Path(path) / "search_state.json")
    return [read_json(Path(path) / "candidates" / f"{h}.json") for h in state["completed"]]


def _new_pending(space, cfg, rng, records, population, count):
    used = {r["hash"] for r in records}
    pending = []
    for _ in range(space.c["max_attempts"]):
        if len(pending) == count:
            break
        if not population or space.e["algorithm"] == "random":
            arch = space.sample(rng)
        else:
            a = Architecture.from_dict(tournament(population, rng)["architecture"])
            b = Architecture.from_dict(tournament(population, rng)["architecture"])
            arch = space.crossover(a, b, rng) if rng.random() < cfg["search"]["crossover_probability"] else a
            if rng.random() < cfg["search"]["mutation_probability"]:
                arch = space.mutate(arch, rng, cfg["search"]["mutation_weights"])
        if arch.hash not in used:
            used.add(arch.hash); pending.append(arch.to_dict())
    if len(pending) != count:
        raise RuntimeError("Unique-candidate generation exhausted max_attempts; reduce budget or enlarge space (no hidden random-immigrant fallback)")
    return pending


def search(cfg, dataset, variant, seed, run_id="main", resume=False, synthetic=False, interrupt_after=None):
    invocation_origin = time.perf_counter()
    if variant not in cfg["experiments"]:
        raise ValueError(f"Unknown experiment: {variant}")
    if cfg["backend"] == "test" and not synthetic:
        raise ValueError("Test backend is permitted only for explicitly synthetic smoke/tests")
    data = DataProtocol(cfg, dataset, synthetic)
    path = run_path(cfg, dataset, variant, seed, run_id)
    identity = {"dataset": dataset, "variant": variant, "search_seed": seed, "run_id": run_id, "synthetic": synthetic}
    with run_lock(path / ".search.lock"):
        init_run(path, cfg, identity, data.manifest, resume)
        return _search_locked(path, cfg, data, variant, seed, interrupt_after, invocation_origin)


def _search_locked(path, cfg, data, variant, seed, interrupt_after, invocation_origin):
    state_path = path / "search_state.json"
    space = Space(cfg["space"], cfg["experiments"][variant])
    upper = cost_upper_bound(cfg["space"], data.max_length, data.channels, len(data.classes))
    cache_protocol = digest({"protocol": protocol_fingerprint(cfg), "data": data.fingerprint, "implementation": implementation_fingerprint()})
    rng = random.Random(seed)
    state = read_json(state_path) if state_path.exists() else {"status": "running", "generation": 0, "completed": [], "population": [], "pending": [], "current_candidate": None, "elapsed_seconds": 0.0, "consecutive_failures": 0, "history": [], "rng": rng.getstate(), "cost_bound": upper, "cache_protocol": cache_protocol}
    if state["cache_protocol"] != cache_protocol or state["cost_bound"] != upper:
        raise ValueError("Search cache protocol mismatch")
    if state["status"] == "complete" and (path / "representatives.json").exists():
        return path
    rng.setstate(tuple_tree(state["rng"]))
    records = [read_json(path / "candidates" / f"{h}.json") for h in state["completed"]]
    indexed = {r["hash"]: r for r in records}
    events = Events(path / "tensorboard" / "search", cfg, len(records))
    invocation_start, previous_elapsed = invocation_origin, state["elapsed_seconds"]
    state["status"] = "running"
    processed = 0

    def save():
        state["elapsed_seconds"] = previous_elapsed + time.perf_counter() - invocation_start
        state["rng"] = rng.getstate()
        atomic_json(state_path, state)

    try:
        while len(records) < cfg["search"]["budget"] or state["pending"]:
            if not state["pending"]:
                population = [indexed[h] for h in state["population"]]
                count = min(cfg["search"]["population"], cfg["search"]["budget"] - len(records))
                state["pending"] = _new_pending(space, cfg, rng, records, population, count)
                save()
            arch = Architecture.from_dict(state["pending"][0])
            h = arch.hash
            state["current_candidate"] = h
            save()
            candidate_path = path / "candidates" / f"{h}.json"
            cache_key = digest({"architecture": h, "evaluation": cache_protocol})
            if candidate_path.exists():
                record = read_json(candidate_path)
                if record["cache_key"] != cache_key:
                    raise ValueError("Candidate cache key mismatch")
            else:
                record = {"hash": h, "architecture": arch.to_dict(), "cache_key": cache_key, "generation": state["generation"], "evaluation_index": len(records) + 1, "macs": macs(arch, data.max_length, data.channels, len(data.classes)), "cost_convention": COST_CONVENTION, "candidate_seed": cfg["seeds"]["candidate"]}
                begin = time.perf_counter()
                record["proxy"] = measure_proxy(arch, data, cfg)
                candidate_events = Events(path / "tensorboard" / "candidates" / h[:12], cfg, 0)
                try:
                    def log_epoch(row):
                        # Per-candidate CSV/JSON is authoritative; interrupted curves
                        # are replaced when the same candidate restarts.
                        for key, value in row.items():
                            if key != "epoch":
                                candidate_events.scalar(key, value, row["epoch"])
                        state["current_epoch"] = row["epoch"]
                        save()
                    trained = train(arch, data, cfg, cfg["seeds"]["candidate"], epoch_callback=log_epoch)
                    record.update({k: trained[k] for k in ("validation", "best_epoch", "epochs_run", "training_seconds", "training_peak_allocated_gpu_bytes", "parameters", "history")})
                    record.update(status="complete", objectives=[1 - trained["validation"]["macro_f1"], math.log1p(record["macs"])], analytic_flops=2 * record["macs"])
                    del trained
                except Exception as exc:
                    record.update(status="failed", error=error_record(exc, "candidate_training"))
                    if str(cfg["device"]).startswith("cuda"):
                        torch.cuda.empty_cache()
                finally:
                    candidate_events.close()
                record["evaluation_seconds"] = time.perf_counter() - begin
                atomic_json(candidate_path, record)
            records.append(record); indexed[h] = record
            state["completed"].append(h)
            state["pending"].pop(0)
            state["current_candidate"] = None
            state["current_epoch"] = None
            state["consecutive_failures"] = state["consecutive_failures"] + 1 if record["status"] == "failed" else 0
            if not state["pending"]:
                pool = [indexed[x] for x in state["population"]] + [r for r in records if r["generation"] == state["generation"] and r["status"] == "complete"]
                pool = list({r["hash"]: r for r in pool}.values())
                state["population"] = [r["hash"] for r in nsga_select(pool, min(len(pool), cfg["search"]["population"]))]
                state["generation"] += 1
            front = pareto(records)
            elapsed = previous_elapsed + time.perf_counter() - invocation_start
            hv = hypervolume([normalized_objective(r, upper) for r in front], cfg["search"]["hv_reference"])
            point = {"candidate_count": len(records), "successful_count": sum(r["status"] == "complete" for r in records), "elapsed_seconds": elapsed, "hv": hv, "pareto_size": len(front), "failure_rate": sum(r["status"] == "failed" for r in records) / len(records), **topology_stats(records)}
            state["history"].append(point)
            for key in ("hv", "pareto_size", "elapsed_seconds", "failure_rate", "topology_count", "topology_entropy"):
                events.scalar("search/" + key, point[key], len(records))
            for scope, values in record["proxy"].get("aggregate", {}).items():
                for key, value in values.items():
                    events.scalar(f"proxy/{scope}/{key}", value, len(records))
            events.scalar("proxy/mean_ce", record["proxy"].get("mean_ce"), len(records))
            save()
            print(f"[{data.name}/{variant}] {len(records)}/{cfg['search']['budget']} {h[:10]} {record['status']} HV={hv:.5f}", flush=True)
            processed += 1
            if state["consecutive_failures"] >= cfg["search"]["consecutive_failure_limit"]:
                raise RuntimeError("Consecutive candidate/backend failures reached limit; inspect candidate errors before resuming")
            if interrupt_after is not None and processed >= interrupt_after:
                raise InterruptedError("Injected test interruption after a committed candidate")
        chosen = representatives(records, upper)
        atomic_json(path / "representatives.json", {"roles": chosen, "architectures": {h: indexed[h]["architecture"] for h in set(chosen.values())}, "rule": "all-time true Pareto; high F1 / fixed-bound normalized distance to ideal / low MACs; deterministic MACs+hash tie break", "cost_upper_bound": upper, "hv_reference": cfg["search"]["hv_reference"]})
        write_csv(path / "search_history.csv", state["history"])
        state.update(status="complete", representatives=chosen)
        save()
        return path
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed"
        state["last_error"] = error_record(exc, "search")
        save()
        raise
    finally:
        events.close()
