"""Three-arm NSGA-II with common scores and isolated mutation RNG streams."""
from collections import defaultdict
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import statistics
import time

from .architecture import Architecture, at, paths
from .artifacts import Events, error_record, init_run, read_json, run_lock
from .config import digest
from .cost import COST_CONVENTION, MULTISCALE_COST_NOTE, cost_upper_bound, macs
from .followup_evaluation import EvaluationCache, atomic_json
from .metrics import hypervolume, normalized_objective, nsga_select, pareto, representatives, topology_stats, tournament
from .search import tuple_tree


def pending_batch(space, cfg, arm, rng, operator_rng, records, population, count, dims, telemetry):
    """B/C have the SAME single-neighbor eligibility gate and leaf slot.

    No extra 1/6 gate. Empty gates consume no operator RNG; legacy fallback
    uses the unchanged base RNG stream. redistribute is never scheduled here.
    """
    from .temporal_ops import neighbors
    used = {r["hash"] for r in records}
    pending = []
    window = cfg["temporal_followup"]["window"]
    for _ in range(space.c["max_attempts"]):
        if len(pending) == count:
            break
        edit = None
        if not population:
            arch = space.sample(rng)
        else:
            a = Architecture.from_dict(tournament(population, rng)["architecture"])
            b = Architecture.from_dict(tournament(population, rng)["architecture"])
            arch = space.crossover(a, b, rng) if rng.random() < cfg["search"]["crossover_probability"] else a
            if rng.random() < cfg["search"]["mutation_probability"]:
                def override(parent):
                    nonlocal edit
                    telemetry["attempts"] += 1
                    single = neighbors(space, parent, "single", dims, window, used)[0]
                    if not single:
                        telemetry["fallback_no_neighbor"] += 1
                        return None
                    options = (single if arm == "single" else
                               neighbors(space, parent, "legacy_leaf", dims, window, used)[0])
                    if not options:
                        raise RuntimeError("Matched control lost a feasible single neighbor")
                    telemetry["eligible"] += 1
                    child = options[operator_rng.choice(sorted(options))]
                    edit = dict(operator=arm, input_hash=parent.hash, input_macs=macs(parent, *dims),
                                child_macs=macs(child, *dims), input_evaluated=parent.hash in {r["hash"] for r in records})
                    return child
                kwargs = {"leaf_override": override} if arm != "original" else {}
                arch = space.mutate(arch, rng, cfg["search"]["mutation_weights"], **kwargs)
        if arch.hash in used:
            continue
        used.add(arch.hash)
        if edit:
            telemetry["submitted"] += 1
        pending.append(dict(architecture=arch.to_dict(), edit=edit))
    if len(pending) != count:
        raise RuntimeError("Unique-candidate generation exhausted max_attempts")
    return pending


def run_search(cfg, dataset, arm, seed, output, interrupt_after=None):
    from .temporal_followup import atomic_csv, data_for, space_for, variant_for
    data = data_for(cfg, dataset)
    if cfg["backend"] == "test" and not data.synthetic:
        raise ValueError("Test mixer is only allowed for synthetic smoke data")
    space = space_for(cfg, dataset, data)
    dims = data.max_length, data.channels, len(data.classes)
    upper = cost_upper_bound(cfg["space"], *dims)
    cache = EvaluationCache(Path(output) / "evaluations", cfg, data)
    path = Path(output) / "search" / dataset / arm / f"seed_{seed}" / "main"
    local = deepcopy(cfg)
    local["followup_arm"] = arm
    local["experiments"] = {arm: deepcopy(cfg["experiments"][variant_for(cfg, dataset)])}
    identity = dict(dataset=dataset, variant=arm, search_seed=seed, run_id="main", synthetic=data.synthetic)
    rng, operator_rng = random.Random(seed), random.Random(digest([seed, "local_edit"]))
    with run_lock(path / ".search.lock"):
        init_run(path, local, identity, data.manifest, (path / "manifest.json").exists())
        state_path = path / "search_state.json"
        state = read_json(state_path) if state_path.exists() else dict(
            status="running", generation=0, completed=[], population=[], pending=[], history=[],
            rng=rng.getstate(), operator_rng=operator_rng.getstate(), elapsed_seconds=0,
            cache_protocol=digest(cache.protocol), cost_bound=upper, evaluation_cost_seconds=0,
            mutation=dict(attempts=0, eligible=0, fallback_no_neighbor=0, submitted=0), consecutive_failures=0)
        if state["cache_protocol"] != digest(cache.protocol) or state["cost_bound"] != upper:
            raise ValueError("Follow-up cache/cost protocol changed")
        if state["status"] == "complete" and (path / "representatives.json").exists():
            print(f"[SKIP] {dataset}/{arm}/{seed}: validated", flush=True)
            return path
        rng.setstate(tuple_tree(state["rng"]))
        operator_rng.setstate(tuple_tree(state["operator_rng"]))
        records = [read_json(path / "candidates" / f"{h}.json") for h in state["completed"]]
        indexed = {r["hash"]: r for r in records}
        events = Events(path / "tensorboard" / "search", cfg, len(records))
        started, previous, processed = time.perf_counter(), state["elapsed_seconds"], 0
        def save():
            state.update(rng=rng.getstate(), operator_rng=operator_rng.getstate(),
                         elapsed_seconds=previous + time.perf_counter() - started)
            atomic_json(state_path, state)
        state["status"] = "running"
        try:
            while len(records) < cfg["search"]["budget"]:
                if not state["pending"]:
                    count = min(cfg["search"]["population"], cfg["search"]["budget"] - len(records))
                    state["pending"] = pending_batch(space, cfg, arm, rng, operator_rng, records,
                                                     [indexed[h] for h in state["population"]], count, dims, state["mutation"])
                    save()
                item = state["pending"][0]
                a = Architecture.from_dict(item["architecture"])
                state["current_candidate"] = a.hash
                save()
                candidate_path = path / "candidates" / f"{a.hash}.json"
                if candidate_path.exists():
                    record = read_json(candidate_path)
                    if record["score_identity"] != digest(cache.identity(a, cfg["seeds"]["candidate"])):
                        raise ValueError("Committed candidate score identity mismatch")
                else:
                    record = dict(hash=a.hash, architecture=a.to_dict(), generation=state["generation"],
                                  evaluation_index=len(records) + 1, macs=macs(a, *dims),
                                  candidate_seed=cfg["seeds"]["candidate"],
                                  score_identity=digest(cache.identity(a, cfg["seeds"]["candidate"])),
                                  cost_convention=COST_CONVENTION + " " + MULTISCALE_COST_NOTE,
                                  scale_shape=a.tree.scale_shape)
                    tokens = math.ceil(data.max_length / (1 if a.tokenizer == "point" else int(a.tokenizer[5:])))
                    record["leaf_scales"] = [dict(path=list(p), rate=n.leaf.rate, input_tokens=tokens,
                                                  scan_tokens=math.ceil(tokens / n.leaf.rate))
                                             for p in paths(a.tree) if (n := at(a.tree, p)).op == "OP"]
                    if item["edit"]:
                        record["temporal_edit"] = item["edit"]
                    candidate_events = Events(path / "tensorboard" / "candidates" / a.hash[:12], cfg)
                    try:
                        def progress(row):
                            state["current_epoch"] = row["epoch"]
                            save()
                        result, meta = cache.measure(a, cfg["seeds"]["candidate"],
                                                     f"{dataset}/{arm}/{seed}", callback=progress)
                        record.update(result, score_cache=meta)
                        record.update(objectives=[1 - result["validation"]["macro_f1"], math.log1p(record["macs"])],
                                      analytic_flops=2 * record["macs"])
                        for row in result["history"]:
                            for key, value in row.items():
                                if key != "epoch":
                                    candidate_events.scalar(key, value, row["epoch"])
                        for scope, values in result["proxy"].get("aggregate", {}).items():
                            for key, value in values.items():
                                if isinstance(value, (int, float)):
                                    candidate_events.scalar(f"proxy/{scope}/{key}", value, 0)
                    except Exception as exc:
                        record.update(status="failed", error=error_record(exc, "followup_candidate"))
                    finally:
                        candidate_events.close()
                    atomic_json(candidate_path, record)
                records.append(record)
                indexed[a.hash] = record
                state["completed"].append(a.hash)
                state["pending"].pop(0)
                state.update(current_candidate=None, current_epoch=None)
                state["consecutive_failures"] = state["consecutive_failures"] + 1 if record["status"] != "complete" else 0
                state["evaluation_cost_seconds"] += record.get("training_seconds", 0) + record.get("proxy", {}).get("seconds", 0)
                if not state["pending"]:
                    pool = [indexed[h] for h in state["population"]] + [r for r in records if r["generation"] == state["generation"] and r["status"] == "complete"]
                    pool = list({r["hash"]: r for r in pool}.values())
                    state["population"] = [r["hash"] for r in nsga_select(pool, min(len(pool), cfg["search"]["population"]))]
                    state["generation"] += 1
                front = pareto(records)
                hv = hypervolume([normalized_objective(r, upper) for r in front], cfg["search"]["hv_reference"])
                point = dict(candidate_count=len(records), successful_count=sum(r["status"] == "complete" for r in records),
                             hv=hv, pareto_size=len(front), elapsed_seconds=previous + time.perf_counter() - started,
                             evaluation_cost_seconds=state["evaluation_cost_seconds"],
                             cache_hits=sum(r.get("score_cache", {}).get("hit", False) for r in records),
                             actual_trainings=sum(r["status"] == "complete" and not r["score_cache"]["hit"] for r in records),
                             failure_rate=sum(r["status"] != "complete" for r in records) / len(records), **topology_stats(records))
                point.update({k.replace("topology_", "scale_structure_"): v for k, v in topology_stats(records, "scale_shape").items()})
                state["history"].append(point)
                for key, value in point.items():
                    if isinstance(value, (int, float)):
                        events.scalar("search/" + key, value, len(records))
                for key, value in state["mutation"].items():
                    events.scalar("mutation/" + key, value, len(records))
                save()
                print(f"[{dataset}/{arm}/{seed}] {len(records)}/{cfg['search']['budget']} {record['status']} HV={hv:.5f}", flush=True)
                processed += 1
                if state["consecutive_failures"] >= cfg["search"]["consecutive_failure_limit"]:
                    raise RuntimeError("Consecutive backend failures; inspect candidate errors")
                if interrupt_after is not None and processed >= interrupt_after:
                    raise InterruptedError("Injected committed-candidate interruption")
            if not any(r["status"] == "complete" for r in records):
                raise RuntimeError("No successful candidates")
            roles = representatives(records, upper)
            atomic_json(path / "representatives.json", dict(roles=roles, architectures={h: indexed[h]["architecture"] for h in set(roles.values())},
                                                           cost_upper_bound=upper, hv_reference=cfg["search"]["hv_reference"]))
            atomic_csv(path / "search_history.csv", state["history"])
            state["status"] = "complete"
            save()
            return path
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
                         last_error=error_record(exc, "followup_search"))
            save()
            raise
        finally:
            events.close()


def run_suite(cfg, datasets, seeds, output, execute, with_refit):
    import json
    from .followup import fit
    from .temporal_followup import data_for
    settings = cfg["temporal_followup"]
    count = len(datasets) * len(seeds) * len(settings["arms"])
    print(json.dumps(dict(datasets=datasets, seeds=seeds, arms=settings["arms"], search_runs=count,
                          candidate_evaluations_upper=count * cfg["search"]["budget"],
                          calibrations_upper=count if with_refit else 0,
                          refit_fits_upper=count * len(cfg["seeds"]["refit"]) if with_refit else 0,
                          output=str(output), with_refit=with_refit, dry_run=not execute), indent=2), flush=True)
    if not execute:
        return
    for dataset in datasets:
        for seed in seeds:
            initial = None
            for arm in settings["arms"]:
                path = run_search(cfg, dataset, arm, seed, output)
                state = read_json(path / "search_state.json")
                records = [read_json(path / "candidates" / f"{h}.json") for h in state["completed"]]
                batch = [(r["hash"], r.get("validation"), r["status"]) for r in records if r["generation"] == 0]
                if initial is not None and batch != initial:
                    raise ValueError(f"Initial architectures OR scores differ: {dataset}/{seed}/{arm}")
                initial = batch
                cap = cfg["temporal_study"]["datasets"][dataset]["cap"]
                eligible = [r for r in records if r["status"] == "complete" and r["macs"] <= cap]
                selected = min(eligible, key=lambda r: (r["objectives"][0], r["macs"], r["hash"]), default=None)
                atomic_json(path / "under_cap.json", dict(cap=cap, cap_hits=len(eligible),
                                                         hash=selected["hash"] if selected else None))
                if with_refit and selected:
                    a = Architecture.from_dict(selected["architecture"])
                    data = data_for(cfg, dataset)
                    # One immutable identity per unique architecture, independent of arm/search seed.
                    # The manifest retains/validates the full hash. A short outer
                    # name avoids duplicating two 64-char hashes on Windows.
                    shared = Path(output) / "shared_refits" / dataset / a.hash[:16]
                    identity = dict(dataset=dataset, variant="shared_under_cap", search_seed=cfg["seeds"]["candidate"],
                                    run_id="shared", synthetic=data.synthetic)
                    # Resume the existing refit implementation after transient
                    # Windows replace locks; committed seeds are already skipped.
                    for attempt in range(8):
                        try:
                            results = list(fit(shared, cfg, data, identity, {"under_cap": a}))
                            break
                        except PermissionError:
                            if os.name != "nt" or attempt == 7:
                                raise
                            time.sleep(0.025 * (attempt + 1))
                    atomic_json(path / "refit_alias.json", dict(shared_path=str(shared.resolve()), hash=a.hash, results=results))
                report(output, write=True)


def report(output, write=False):
    import json
    from .temporal_followup import atomic_csv
    output = Path(output)
    rows, histories, fits, failures = [], [], [], []
    for state_path in sorted((output / "search").glob("*/*/seed_*/main/search_state.json")):
        path = state_path.parent
        manifest = read_json(path / "manifest.json")
        state = read_json(state_path)
        info = manifest["identity"]
        group = dict(dataset=info["dataset"], arm=info["variant"], search_seed=info["search_seed"])
        hist = state["history"]
        histories.extend(dict(group, **r) for r in hist)
        row = dict(group, status=state["status"], candidates=len(state["completed"]), current_candidate=state.get("current_candidate"),
                   current_epoch=state.get("current_epoch"), elapsed_seconds=state["elapsed_seconds"],
                   **state["mutation"])
        if hist:
            row.update(hv=hist[-1]["hv"], pareto_size=hist[-1]["pareto_size"], cache_hits=hist[-1]["cache_hits"],
                       actual_trainings=hist[-1]["actual_trainings"], evaluation_cost_seconds=hist[-1]["evaluation_cost_seconds"],
                       failure_rate=hist[-1]["failure_rate"],
                       hv_auc_per_eval=sum(h["hv"] for h in hist[:-1]) / len(hist))
        if (path / "under_cap.json").exists():
            selected = read_json(path / "under_cap.json")
            row.update(selected)
            if selected["hash"]:
                r = read_json(path / "candidates" / f"{selected['hash']}.json")
                row.update(validation_f1=r["validation"]["macro_f1"], macs=r["macs"])
        if (path / "refit_alias.json").exists():
            alias = read_json(path / "refit_alias.json")
            rr = alias["results"]
            fits.extend(dict(r, **group, shared_path=alias["shared_path"]) for r in rr)
            vals = [r["macro_f1"] for r in rr]
            row.update(test_f1=statistics.mean(vals), refit_internal_sd=statistics.stdev(vals) if len(vals) > 1 else None)
        if state.get("last_error") and state["status"] != "complete":
            failures.append(dict(group, error=state["last_error"]))
        rows.append(row)
    summary, paired = [], []
    for d, arm in sorted({(r["dataset"], r["arm"]) for r in rows}):
        rr = [r for r in rows if (r["dataset"], r["arm"]) == (d, arm) and r["status"] == "complete"]
        result = dict(dataset=d, arm=arm, completed_searches=len(rr),
                      searches_hit=sum(bool(r.get("hash")) for r in rr))
        for field in ("hv", "hv_auc_per_eval", "validation_f1", "test_f1", "macs"):
            values = [r[field] for r in rr if field in r]
            result.update({field + "_n": len(values), field + "_mean": statistics.mean(values) if values else None,
                           field + "_sd_between_searches": statistics.stdev(values) if len(values) > 1 else None})
        summary.append(result)
    for d, seed in sorted({(r["dataset"], r["search_seed"]) for r in rows}):
        rr = {r["arm"]: r for r in rows if (r["dataset"], r["search_seed"]) == (d, seed) and r["status"] == "complete"}
        for left, right in (("original", "single"), ("matched_random", "single"), ("original", "matched_random")):
            if left not in rr or right not in rr:
                continue
            result = dict(dataset=d, search_seed=seed, control=left, proposed=right)
            for field, mult in (("hv", 1), ("hv_auc_per_eval", 1), ("validation_f1", 100), ("test_f1", 100)):
                result[field + "_delta" + ("_pp" if mult == 100 else "")] = mult * (rr[right][field] - rr[left][field]) if field in rr[left] and field in rr[right] else None
            paired.append(result)
    snapshot = dict(searches=len(rows), complete=sum(r["status"] == "complete" for r in rows),
                    unique_refit_fits=len({(r["shared_path"], r["seed"]) for r in fits}), failures=failures,
                    synthetic_warning="CPU test mixer: workflow verification only" if any(read_json(p)["config"]["backend"] == "test" for p in (output / "search").glob("*/*/seed_*/main/manifest.json")) else None,
                    statistical_unit="search seed; refit seeds nested; shared architectures are aliases",
                    cache_note="Wall time includes reuse; compare logical evaluations and attributed evaluation cost separately",
                    summary=summary)
    if (output / "audit" / "plan.json").exists():
        jobs = read_json(output / "audit" / "plan.json")
        snapshot["audit"] = dict(planned=len(jobs), complete=sum(Path(j["result_path"]).exists() for j in jobs),
                                  comparisons=str(output / "audit" / "comparisons.csv"))
    if (output / "diagnose" / "plan.json").exists():
        plans = read_json(output / "diagnose" / "plan.json")
        snapshot["diagnose"] = {d: dict(pairable_parents=len(p["parents"]), shortage=p["shortage"]) for d, p in plans.items()}
    snapshot["census_complete"] = (output / "census" / "summary.csv").exists()
    if write:
        for name, content in (("search_runs", rows), ("search_hv", histories), ("search_refits", fits),
                              ("search_summary", summary), ("search_paired", paired)):
            atomic_csv(output / "reports" / f"{name}.csv", content)
        atomic_json(output / "reports" / "status.json", snapshot)
    print(json.dumps(snapshot, indent=2), flush=True)
    return snapshot
