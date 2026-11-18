"""Opt-in follow-up experiments: audit, census, diagnosis and paired search.

Every command previews by default. Only search --with-refit may load TEST.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

import yaml

from .architecture import Architecture, Space, at, paths
from .artifacts import Events, environment, error_record, read_json, run_lock, write_csv
from .config import DEFAULT_CONFIG, digest, load_config, merge, resolve_path, validate
from .cost import macs
from .followup_evaluation import EvaluationCache, atomic_json, fixed_train, source_identity
from .temporal_ops import neighbors


def configuration(path=None, profile="paper", overrides=()):
    cfg = load_config(path, profile, multiscale=True, temporal=True)
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))["temporal_followup"]
    settings = {k: v for k, v in raw.items() if k != "profiles"}
    cfg["temporal_followup"] = merge(settings, raw["profiles"][profile])
    cfg["search"].update(cfg["temporal_followup"]["search"])
    if profile == "smoke":
        cfg["smoke"].update(cfg["temporal_followup"]["smoke_data"])
    for value in overrides:
        key, sep, content = value.partition("=")
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown --set path: {key}")
            target = target[part]
        if not sep or parts[-1] not in target:
            raise ValueError(f"Unknown --set path: {key}")
        target[parts[-1]] = yaml.safe_load(content)
        if key.startswith("temporal_followup.search."):
            cfg["search"][parts[-1]] = target[parts[-1]]
        if profile == "smoke" and key.startswith("temporal_followup.smoke_data."):
            cfg["smoke"][parts[-1]] = target[parts[-1]]
    settings = cfg["temporal_followup"]
    if not 0 <= settings["window"] <= 1 or not settings["census_windows"] or any(not 0 <= x <= 1 for x in settings["census_windows"]):
        raise ValueError("Invalid follow-up cost windows")
    for section, keys in (("audit", ("architectures", "repeats", "epochs")),
                          ("diagnostic", ("parents", "children", "epochs"))):
        if any(type(settings[section][k]) is not int or settings[section][k] < 1 for k in keys):
            raise ValueError(f"Invalid {section} budgets")
    if settings["audit"]["repeats"] < 2:
        raise ValueError("Audit requires at least two independent repeats")
    for seeds in (settings["search_seeds"], settings["audit"]["seeds"], settings["diagnostic"]["seeds"]):
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
            raise ValueError("Follow-up seeds must be unique nonempty integers")
    if not settings["source_seeds"] or len(set(settings["source_seeds"])) != len(settings["source_seeds"]):
        raise ValueError("Source seeds must be nonempty and unique")
    if not settings["datasets"] or any(d not in cfg["temporal_study"]["datasets"] for d in settings["datasets"]):
        raise ValueError("Unsupported follow-up datasets")
    if settings["arms"] != ["original", "matched_random", "single"]:
        raise ValueError("The controlled comparison requires original/matched_random/single")
    validate(cfg)
    return cfg


def variant_for(cfg, dataset):
    return "tree_multiscale_" + cfg["temporal_study"]["datasets"][dataset]["direction"]


def data_for(cfg, dataset):
    from .data import DataProtocol
    return DataProtocol(cfg, dataset, cfg["profile"] == "smoke")


def space_for(cfg, dataset, data):
    return Space(cfg["space"], cfg["experiments"][variant_for(cfg, dataset)],
                 min(len(data.samples[i]) for i in data.train_indices))


def source_pool(cfg, dataset, root):
    """Read TRAIN candidate scores only; never read refit or TEST tables."""
    data = data_for(cfg, dataset)
    space = space_for(cfg, dataset, data)
    found, sources = {}, []
    if cfg["profile"] == "smoke":
        rng = random.Random(cfg["temporal_followup"]["sampling_seed"])
        # Fixed sampling seed and a finite fixture; production uses saved sources.
        for _ in range(cfg["space"]["max_attempts"]):
            a = space.sample(rng)
            found[a.hash] = {"architecture": a.to_dict(), "hash": a.hash, "validation_f1": 0.5,
                             "source_seed": cfg["data"]["split_seed"]}
            if len(found) >= cfg["search"]["population"] * 8:
                break
    else:
        for manifest_path in sorted((Path(root) / dataset).glob("*/seed_*/*/manifest.json")):
            m = read_json(manifest_path)
            identity = m["identity"]
            if identity.get("search_seed") not in cfg["temporal_followup"]["source_seeds"]:
                continue
            exp = m["config"]["experiments"][identity["variant"]]
            wanted = cfg["experiments"][variant_for(cfg, dataset)]
            if exp != wanted:
                continue
            state_path = manifest_path.parent / "search_state.json"
            if not state_path.exists() or (state := read_json(state_path))["status"] != "complete":
                continue
            if m["data"] != data.manifest:
                raise ValueError(f"Source TRAIN/split mismatch: {manifest_path}")
            sources.append(str(manifest_path.resolve()))
            for h in state["completed"]:
                r = read_json(manifest_path.parent / "candidates" / f"{h}.json")
                if r["status"] != "complete":
                    continue
                a = Architecture.from_dict(r["architecture"])
                if a.hash != h or not space.valid(a) or r["macs"] != macs(a, data.max_length, data.channels, len(data.classes)):
                    raise ValueError(f"Invalid saved candidate: {h}")
                # Avoid selecting the best of repeated noisy scores of one hash.
                found.setdefault(h, dict(hash=h, architecture=a.to_dict(),
                                         validation_f1=r["validation"]["macro_f1"],
                                         source_seed=identity["search_seed"]))
        if not found:
            raise ValueError(f"No completed matching multiscale TRAIN candidates: {Path(root) / dataset}")
    return data, space, list(found.values()), sources


def stratified_order(rows, sampling_seed):
    strata = defaultdict(list)
    for row in rows:
        a = Architecture.from_dict(row["architecture"])
        strata[(a.tokenizer, a.tree.leaves, a.tree.shape)].append(row)
    groups = []
    for key in sorted(strata):
        groups.append(sorted(strata[key], key=lambda r: digest([sampling_seed, r["hash"]])))
    return [g[i] for i in range(max(map(len, groups), default=0)) for g in groups if i < len(g)]


def bind(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Protocol changed; use a new --output directory: {path}")
    else:
        atomic_json(path, value)


def atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(".csv.tmp")
    write_csv(temporary, rows)
    os.replace(temporary, path)


def census(cfg, datasets, source, output, execute):
    rows, plans = [], {}
    settings = cfg["temporal_followup"]
    for dataset in datasets:
        data, space, pool, sources = source_pool(cfg, dataset, source)
        plans[dataset] = {"data": data.manifest, "sources": sources,
                          "parents": [r["hash"] for r in pool]}
        dims = data.max_length, data.channels, len(data.classes)
        for r in pool:
            a = Architecture.from_dict(r["architecture"])
            for window in settings["census_windows"]:
                for proposed, control in (("single", "legacy_leaf"), ("redistribute", "random_double")):
                    aa, ac = neighbors(space, a, proposed, dims, window)
                    bb, bc = neighbors(space, a, control, dims, window)
                    sa, sb = set(aa), set(bb)
                    rows.append(dict(dataset=dataset, parent=a.hash, tokenizer=a.tokenizer,
                                     shape=a.tree.shape, scale_shape=a.tree.scale_shape, window=window,
                                     proposed=proposed, control=control, cost_scope="total" if proposed == "single" else "trunk",
                                     proposed_count=len(sa), control_count=len(sb), shared_count=len(sa & sb),
                                     jaccard=len(sa & sb) / len(sa | sb) if sa | sb else None,
                                     pairable=bool(sa and sb), identical=bool(sa) and sa == sb,
                                     proposed_rejections=ac, control_rejections=bc))
    summary = []
    for key in sorted({(r["dataset"], r["window"], r["proposed"]) for r in rows}):
        rr = [r for r in rows if (r["dataset"], r["window"], r["proposed"]) == key]
        summary.append(dict(zip(("dataset", "window", "proposed"), key), parents=len(rr),
                            covered=sum(r["proposed_count"] > 0 for r in rr),
                            pairable=sum(r["pairable"] for r in rr), identical=sum(r["identical"] for r in rr)))
    print(json.dumps({"census": summary, "output": str(output), "dry_run": not execute}, indent=2))
    if execute:
        bind(output / "census" / "plan.json", plans)
        atomic_csv(output / "census" / "neighborhoods.csv", rows)
        atomic_csv(output / "census" / "summary.csv", summary)


def diagnostic_plan(cfg, data, space, pool):
    settings = cfg["temporal_followup"]
    dims = data.max_length, data.channels, len(data.classes)
    selected, rejected = [], Counter()
    for row in stratified_order(pool, settings["sampling_seed"]):
        a = Architecture.from_dict(row["architecture"])
        single = neighbors(space, a, "single", dims, settings["window"])[0]
        control = neighbors(space, a, "legacy_leaf", dims, settings["window"])[0]
        if not single or not control:
            rejected["no_pair"] += 1
            continue
        if set(single) == set(control):
            rejected["identical_neighborhood"] += 1
            continue
        count = min(settings["diagnostic"]["children"], len(single), len(control))
        def sample(options):
            order = sorted(options, key=lambda h: digest([settings["sampling_seed"], a.hash, h]))
            return [options[h].to_dict() for h in order[:count]]
        children = {"single": sample(single), "matched_random": sample(control)}
        if children["single"] == children["matched_random"]:
            rejected["identical_sample"] += 1
            continue
        selected.append(dict(parent=a.to_dict(), children=children, quota=count))
        if len(selected) == settings["diagnostic"]["parents"]:
            break
    return selected, dict(rejected)


def diagnose(cfg, datasets, source, output, execute):
    settings = cfg["temporal_followup"]["diagnostic"]
    plans, contexts = {}, {}
    for dataset in datasets:
        data, space, pool, sources = source_pool(cfg, dataset, source)
        selected, rejected = diagnostic_plan(cfg, data, space, pool)
        plans[dataset] = dict(data=data.manifest, sources=sources, parents=selected, rejected=rejected,
                              requested=settings["parents"], shortage=settings["parents"] - len(selected))
        contexts[dataset] = data
    upper = sum(sum(1 + 2 * r["quota"] for r in p["parents"]) for p in plans.values()) * len(settings["seeds"])
    print(json.dumps({"parents": {d: len(p["parents"]) for d, p in plans.items()}, "training_upper": upper,
                      "shortage": {d: p["shortage"] for d, p in plans.items()}, "dry_run": not execute}, indent=2))
    if not execute:
        return
    bind(output / "diagnose" / "plan.json", plans)
    rows, comparisons = [], []
    for dataset, plan in plans.items():
        cache = EvaluationCache(output / "evaluations", cfg, contexts[dataset])
        for entry in plan["parents"]:
            parent = Architecture.from_dict(entry["parent"])
            for seed in settings["seeds"]:
                pv, _ = cache.measure(parent, seed, "diagnostic_parent", "diagnostic", settings["epochs"])
                log_diagnostic(output, cfg, dataset, parent.hash, seed, pv)
                values = defaultdict(list)
                for arm, children in entry["children"].items():
                    for child in children:
                        a = Architecture.from_dict(child)
                        result, meta = cache.measure(a, seed, f"diagnostic_{arm}", "diagnostic", settings["epochs"])
                        log_diagnostic(output, cfg, dataset, a.hash, seed, result)
                        for kind, field in (("search_best", "validation"), ("fixed_end", "final_validation")):
                            delta = 100 * (result[field]["macro_f1"] - pv[field]["macro_f1"])
                            values[arm, kind].append(delta)
                            rows.append(dict(dataset=dataset, parent=parent.hash, child=a.hash, arm=arm, seed=seed,
                                             protocol=kind, f1_delta_pp=delta, child_f1=result[field]["macro_f1"],
                                             parent_f1=pv[field]["macro_f1"], cache_hit=meta["hit"]))
                for kind in ("search_best", "fixed_end"):
                    prop, ctrl = statistics.mean(values["single", kind]), statistics.mean(values["matched_random", kind])
                    comparisons.append(dict(dataset=dataset, parent=parent.hash, seed=seed, protocol=kind,
                                            single_delta_pp=prop, control_delta_pp=ctrl, paired_delta_pp=prop - ctrl))
                atomic_csv(output / "diagnose" / "children.csv", rows)
                atomic_csv(output / "diagnose" / "parent_seed_pairs.csv", comparisons)
                print(f"[DIAGNOSE] {dataset} {parent.hash[:10]} seed={seed}", flush=True)
    parents, summary = [], []
    for d, h, kind in sorted({(r["dataset"], r["parent"], r["protocol"]) for r in comparisons}):
        rr = [r for r in comparisons if (r["dataset"], r["parent"], r["protocol"]) == (d, h, kind)]
        parents.append(dict(dataset=d, parent=h, protocol=kind, seeds=len(rr),
                            **{k: statistics.mean(r[k] for r in rr) for k in ("single_delta_pp", "control_delta_pp", "paired_delta_pp")}))
    for d, kind in sorted({(r["dataset"], r["protocol"]) for r in parents}):
        rr = [r for r in parents if (r["dataset"], r["protocol"]) == (d, kind)]
        diffs = [r["paired_delta_pp"] for r in rr]
        summary.append(dict(dataset=d, protocol=kind, parents=len(rr), mean_delta_pp=statistics.mean(diffs),
                            sd_delta_pp=statistics.stdev(diffs) if len(diffs) > 1 else None,
                            wins=sum(x > 1e-9 for x in diffs), losses=sum(x < -1e-9 for x in diffs)))
    atomic_csv(output / "diagnose" / "parents.csv", parents)
    atomic_csv(output / "diagnose" / "summary.csv", summary)


def log_diagnostic(output, cfg, dataset, h, seed, result):
    events = Events(output / "diagnose" / "tensorboard" / dataset / h[:12] / str(seed), cfg, 0)
    try:
        for row in result["history"]:
            for key in ("train_ce", "validation_ce", "validation_macro_f1"):
                events.scalar(key, row[key], row["epoch"])
    finally:
        events.close()


def audit_worker(job_path):
    job = read_json(job_path)
    result_path = Path(job["result_path"])
    try:
        data = data_for(job["config"], job["dataset"])
        if data.manifest != job["data"]:
            raise ValueError("Audit worker TRAIN/split differs from planned input")
        events = Events(result_path.parent / "tensorboard" / result_path.stem, job["config"])
        try:
            def log(row):
                for key in ("train_ce", "validation_ce", "validation_macro_f1"):
                    events.scalar(key, row[key], row["epoch"])
            result = fixed_train(Architecture.from_dict(job["architecture"]), data, job["config"],
                                 job["seed"], job["epochs"], log)
        finally:
            events.close()
        atomic_json(result_path, dict(result, environment=environment(), pid=os.getpid(),
                                      job_fingerprint=digest(job)))
    except Exception as exc:
        atomic_json(result_path.with_suffix(".error.json"), error_record(exc, "audit_worker"))
        raise


def audit(cfg, datasets, source, output, execute):
    settings = cfg["temporal_followup"]["audit"]
    jobs = []
    for dataset in datasets:
        data, _, pool, _ = source_pool(cfg, dataset, source)
        dims = data.max_length, data.channels, len(data.classes)
        ranked = sorted(pool, key=lambda r: (-r["validation_f1"], r["hash"]))
        cap = cfg["temporal_study"]["datasets"][dataset]["cap"]
        eligible = [r for r in ranked if macs(Architecture.from_dict(r["architecture"]), *dims) <= cap]
        first = (eligible or ranked)[0]
        rest = sorted((r for r in pool if r["hash"] != first["hash"]),
                      key=lambda r: (macs(Architecture.from_dict(r["architecture"]), *dims), r["hash"]))
        selected = [first] + rest[:settings["architectures"] - 1]
        for row in selected:
            for seed in settings["seeds"]:
                for repeat in range(settings["repeats"]):
                    dest = output / "audit" / dataset / row["hash"] / f"seed_{seed}_repeat_{repeat}.json"
                    jobs.append(dict(config=cfg, data=data.manifest, dataset=dataset, architecture=row["architecture"],
                                     seed=seed, repeat=repeat, epochs=settings["epochs"], result_path=str(dest.resolve())))
    print(json.dumps({"independent_process_fits": len(jobs), "epochs": settings["epochs"], "dry_run": not execute}, indent=2))
    if not execute:
        return
    bind(output / "audit" / "plan.json", jobs)
    for job in jobs:
        result_path = Path(job["result_path"])
        if result_path.exists():
            saved = read_json(result_path)
            if saved.get("status") != "complete" or saved.get("job_fingerprint") != digest(job):
                raise ValueError(f"Audit result protocol mismatch: {result_path}")
            print(f"[SKIP] audit {result_path}", flush=True)
            continue
        job_path = result_path.with_suffix(".job.json")
        atomic_json(job_path, job)
        print(f"[AUDIT] {job['dataset']} seed={job['seed']} repeat={job['repeat']}", flush=True)
        subprocess.run([sys.executable, "-m", "mamba_tree_nas.temporal_followup", "_audit-worker", str(job_path)],
                       check=True, cwd=str(Path(__file__).resolve().parent.parent))
    comparison = []
    by_key = defaultdict(list)
    for job in jobs:
        by_key[job["dataset"], Architecture.from_dict(job["architecture"]).hash, job["seed"]].append(job)
    for (dataset, h, seed), group in by_key.items():
        base = read_json(group[0]["result_path"])
        for job in group[1:]:
            other = read_json(job["result_path"])
            first_difference = "initial_weights" if base["initial_weights_hash"] != other["initial_weights_hash"] else None
            for left, right in zip(base["history"], other["history"]):
                for field in ("sample_order_hash", "first_batch_ids_hash", "first_logits_hash", "first_loss",
                              "first_gradient_hash", "first_update_hash", "validation_macro_f1"):
                    if left[field] != right[field] and first_difference is None:
                        first_difference = f"epoch={left['epoch']}:{field}"
            if first_difference is None and base["final_weights_hash"] != other["final_weights_hash"]:
                first_difference = "later_batch_or_final_weights"
            comparison.append(dict(dataset=dataset, hash=h, seed=seed, repeat=job["repeat"],
                                   initial_equal=base["initial_weights_hash"] == other["initial_weights_hash"],
                                   final_equal=base["final_weights_hash"] == other["final_weights_hash"],
                                   first_difference=first_difference,
                                   best_f1_delta_pp=100 * (other["validation"]["macro_f1"] - base["validation"]["macro_f1"]),
                                   final_f1_delta_pp=100 * (other["final_validation"]["macro_f1"] - base["final_validation"]["macro_f1"])))
    atomic_csv(output / "audit" / "comparisons.csv", comparison)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "_audit-worker":
        audit_worker(Path(argv[1]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "census", "diagnose", "search", "report", "status"))
    parser.add_argument("--profile", choices=("paper", "pilot", "smoke"), default="paper")
    parser.add_argument("--dataset", action="append", choices=("Heartbeat", "PEMS-SF", "SpokenArabicDigits"))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--source-root", type=Path, default=Path("mamba_tree_nas/work/temporal_mutation/search/paper"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-refit", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    if args.seed and (len(set(args.seed)) != len(args.seed) or args.stage != "search"):
        parser.error("--seed is a unique search-seed subset, only for search")
    if args.with_refit and args.stage != "search":
        parser.error("--with-refit is only valid for search")
    cfg = configuration(args.config, args.profile, args.overrides)
    if args.data_root:
        cfg["data"]["root"] = str(args.data_root)
    output = resolve_path(args.output or Path("mamba_tree_nas/work/temporal_followup") / args.profile).resolve()
    source = resolve_path(args.source_root).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Output must be separate from the read-only source results")
    datasets = list(dict.fromkeys(args.dataset or cfg["temporal_followup"]["datasets"]))
    if args.seed and any(s not in cfg["temporal_followup"]["search_seeds"] for s in args.seed):
        parser.error("--seed must belong to temporal_followup.search_seeds; use --set to configure new seeds")
    protocol = {"config": cfg, "source": str(source), "environment": environment(),
                "source_identity": source_identity(),
                "entry": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "search_entry": hashlib.sha256(Path(__file__).with_name("followup_search.py").read_bytes()).hexdigest(),
                "operator_entry": hashlib.sha256(Path(__file__).with_name("temporal_ops.py").read_bytes()).hexdigest(),
                "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if args.stage in ("status", "report"):
        from .followup_search import report
        report(output, write=args.stage == "report" and args.execute)
        return
    if args.execute:
        with run_lock(output / ".study.lock"):
            bind(output / "protocol.json", protocol)
            dispatch(args, cfg, datasets, source, output)
    else:
        dispatch(args, cfg, datasets, source, output)


def dispatch(args, cfg, datasets, source, output):
    if args.stage == "search":
        from .followup_search import run_suite
        run_suite(cfg, datasets, args.seed or cfg["temporal_followup"]["search_seeds"], output,
                  args.execute, args.with_refit)
    else:
        {"audit": audit, "census": census, "diagnose": diagnose}[args.stage](cfg, datasets, source, output, args.execute)


if __name__ == "__main__":
    main()
