"""Validation-only mutation diagnostics and paired NSGA-II experiments."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import re
import statistics

from .architecture import Architecture, Leaf, Node, Space, at, paths
from .artifacts import atomic_json, implementation_fingerprint, read_json, run_lock, write_csv
from .config import digest, load_config, resolve_path
from .cost import macs, trunk_macs
from .temporal_ops import neighbors


def variant_for(cfg, dataset):
    return "tree_multiscale_" + cfg["temporal_study"]["datasets"][dataset]["direction"]


def dimensions(data):
    return data.max_length, data.channels, len(data.classes)


def make_space(cfg, variant, data):
    return Space(cfg["space"], cfg["experiments"][variant],
                 min(len(data.samples[i]) for i in data.train_indices))


def _source_parents(args, cfg, dataset, variant):
    from .search import load_records, run_path
    from .data import DataProtocol

    base = {"output": str(resolve_path(args.source_root).resolve())}
    groups, reference, data = [], None, None
    for seed in cfg["seeds"]["search"]:
        path = run_path(base, dataset, variant, seed, args.source_run_id)
        m = read_json(path / "manifest.json")
        expected = dict(dataset=dataset, variant=variant, search_seed=seed, run_id=args.source_run_id, synthetic=False)
        if m["identity"] != expected or read_json(path / "search_state.json")["status"] != "complete":
            raise ValueError(f"Source identity mismatch or incomplete search: {path}")
        saved = m["config"]
        fields = ("backend", "device", "runtime", "space", "train")
        protocol = {k: saved[k] for k in fields}
        protocol.update(experiment=saved["experiments"][variant], candidate_seed=saved["seeds"]["candidate"],
                        data=m["data"])
        if reference is None:
            reference = protocol
            local = deepcopy(saved)
            if args.data_root is not None:
                local["data"]["root"] = str(args.data_root)
            data = DataProtocol(local, dataset, False)
            if data.manifest != m["data"]:
                raise ValueError(f"Live TRAIN source or split differs from {path}")
        elif protocol != reference:
            raise ValueError(f"Incompatible source training protocol: {path}")
        rows = []
        for r in load_records(path):
            if r["status"] != "complete":
                continue
            a = Architecture.from_dict(r["architecture"])
            if r["hash"] != a.hash or r["macs"] != macs(a, *dimensions(data)) or r["candidate_seed"] != protocol["candidate_seed"]:
                raise ValueError(f"Source candidate metadata mismatch: {path}/{r['hash']}")
            rows.append((seed, r["validation"]["macro_f1"], a))
        groups.append(rows)
    return local, data, groups


def _synthetic_parents(cfg, dataset, variant):
    from .data import DataProtocol

    local = deepcopy(cfg)
    # Give the CPU fixture a genuine scale neighborhood and a large input
    # projection, so the single-edit cost screen also has admissible children.
    local["smoke"].update(length=261, channels=48)
    data = DataProtocol(local, dataset, True)
    leaf = lambda r: Node("OP", leaf=Leaf("F", 2, 2, 1, r))
    trees = ([leaf(1), Node("RES", (leaf(2),))] if dataset == "PEMS-SF" else
             [Node("PAR", (leaf(2), Node("RES", (leaf(4),)))),
              Node("PAR", (leaf(1), Node("RES", (leaf(2),))))])
    return local, data, [[(2021, 0.5, Architecture("patch8", 8, "mean", tree, 2)) for tree in trees]]


def diagnostic_plan(groups, space, dataset, dims, settings, window):
    proposed, control = ("single", "legacy_leaf") if dataset == "PEMS-SF" else ("redistribute", "random_double")
    def structural(a):
        return a.tree.leaves == 1 if dataset == "PEMS-SF" else any(at(a.tree, p).op == "PAR" for p in paths(a.tree))

    ranked = [sorted((x for x in g if structural(x[2])),
                     key=lambda x: (-x[1], macs(x[2], *dims), x[2].hash)) for g in groups]
    seen, census, selected = set(), [], []
    for i in range(max(map(len, ranked), default=0)):
        for group in ranked:
            if i >= len(group):
                continue
            seed, f1, parent = group[i]
            if parent.hash in seen:
                continue
            seen.add(parent.hash)
            if not space.valid(parent):
                raise ValueError(f"Ineligible saved architecture: {parent.hash}")
            neighborhoods = {m: neighbors(space, parent, m, dims, window)[0] for m in (proposed, control)}
            a, b = map(set, neighborhoods.values())
            row = dict(dataset=dataset, parent=parent.hash, source_seed=seed, source_val_f1=f1,
                       proposed=proposed, control=control, proposed_count=len(a), control_count=len(b),
                       shared_count=len(a & b), shared_fraction=len(a & b) / len(a | b) if a or b else None,
                       identical=bool(a) and a == b)
            census.append(row)
            if len(selected) >= settings["parents"]:
                continue
            quota = min(settings["children_per_operator"], len(a), len(b))
            entry = dict(row, quota=quota, architecture=parent.to_dict(), children={})
            picked = {}
            for m, options in neighborhoods.items():
                # Common seeded ordering means identical neighborhoods select identical children.
                order = sorted(options, key=lambda h: digest([settings["sampling_seed"], parent.hash, h]))
                picked[m] = order[:quota]
                entry["children"][m] = [options[h].to_dict() for h in picked[m]]
            entry["same_sample"] = quota > 0 and picked[proposed] == picked[control]
            selected.append(entry)
    return census, selected


def _diagnostic_measure(directory, arch, data, cfg, protocol, fixed_epochs):
    from .training import evaluate, train

    key = digest({"hash": arch.hash, "protocol": protocol, "fixed_epochs": fixed_epochs})
    path = directory / "cache" / f"{key}.json"
    if path.exists():
        return read_json(path)
    result = train(arch, data, cfg, cfg["seeds"]["candidate"], fixed_epochs=fixed_epochs)
    val = result["validation"]
    if fixed_epochs is not None:
        val, _ = evaluate(result["model"], data.val, cfg, len(data.classes))
    row = {"validation_f1": val["macro_f1"], "epoch": result["best_epoch"],
           "training_seconds": result["training_seconds"]}
    atomic_json(path, row)
    return row


def bind(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Protocol changed; use a new --output: {path}")
    else:
        atomic_json(path, value)


def study_protocol():
    import hashlib

    return {"core": implementation_fingerprint(), "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def diagnose(args, cfg, datasets, output):
    census, selected, contexts = [], [], {}
    synthetic = cfg["profile"] == "smoke"
    policy = cfg["search"]["temporal_mutation"]
    settings = deepcopy(cfg["temporal_study"]["diagnostics"])
    if synthetic:
        settings.update(fixed_parents=1, fixed_epochs=2)
    for dataset in datasets:
        if dataset not in ("Heartbeat", "PEMS-SF"):
            raise ValueError("Diagnostics are defined for Heartbeat and PEMS-SF")
        variant = variant_for(cfg, dataset)
        local, data, groups = (_synthetic_parents(cfg, dataset, variant) if synthetic
                               else _source_parents(args, cfg, dataset, variant))
        dims = dimensions(data)
        counts, plans = diagnostic_plan(groups, make_space(local, variant, data), dataset, dims, settings, policy["window"])
        census.extend(counts)
        chosen_fixed = [x["parent"] for x in plans if x["quota"]][:settings["fixed_parents"]]
        selected.extend(dict(x, fixed_epochs=settings["fixed_epochs"] if x["parent"] in chosen_fixed else None) for x in plans)
        contexts[dataset] = local, data, dims
    upper = sum((1 + 2 * r["quota"]) * (1 + bool(r["fixed_epochs"])) for r in selected if r["quota"])
    print(json.dumps({"output": str(output), "parents": selected, "training_upper": upper, "dry_run": not args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    with run_lock(output / ".study.lock"):
        protocols = {d: digest([c, data.manifest, study_protocol()]) for d, (c, data, _) in contexts.items()}
        bind(output / "diagnostic_plan.json", dict(settings=settings, window=policy["window"],
                                                  protocols=protocols, census=census, parents=selected))
        write_csv(output / "diagnostic_coverage.csv", census)
        write_csv(output / "diagnostic_coverage_summary.csv", [
            dict(dataset=d, parents=len(rows), proposed_covered=sum(r["proposed_count"] > 0 for r in rows),
                 control_covered=sum(r["control_count"] > 0 for r in rows),
                 pairable=sum(bool(r["proposed_count"] and r["control_count"]) for r in rows),
                 identical=sum(r["identical"] for r in rows))
            for d in datasets for rows in [[r for r in census if r["dataset"] == d]]])
        child_rows, parent_rows = [], []
        for entry in selected:
            if not entry["quota"]:
                continue
            d = entry["dataset"]
            local, data, dims = contexts[d]
            parent = Architecture.from_dict(entry["architecture"])
            for epochs in [None] + ([entry["fixed_epochs"]] if entry["fixed_epochs"] else []):
                kind = "fixed_end" if epochs is not None else "search_best"
                measure = lambda a: _diagnostic_measure(output, a, data, local, protocols[d], epochs)
                pv = measure(parent)
                deltas = {}
                for mode, children in entry["children"].items():
                    deltas[mode] = []
                    for child in map(Architecture.from_dict, children):
                        cv = measure(child)
                        delta = 100 * (cv["validation_f1"] - pv["validation_f1"])
                        deltas[mode].append(delta)
                        child_rows.append(dict(dataset=d, parent=parent.hash, child=child.hash, mode=mode,
                                               protocol=kind, parent_f1=pv["validation_f1"], child_f1=cv["validation_f1"],
                                               parent_epoch=pv["epoch"], child_epoch=cv["epoch"], f1_delta_pp=delta,
                                               macs=macs(child, *dims), macs_delta=macs(child, *dims)-macs(parent, *dims),
                                               trunk_delta=trunk_macs(child, dims[0])-trunk_macs(parent, dims[0])))
                prop, ctrl = (statistics.mean(deltas[entry[k]]) for k in ("proposed", "control"))
                parent_rows.append(dict(dataset=d, parent=parent.hash, protocol=kind, identical=entry["identical"],
                                        same_sample=entry["same_sample"],
                                        proposed_delta_pp=prop, control_delta_pp=ctrl, paired_delta_pp=prop-ctrl,
                                        proposed_improvement_fraction=statistics.mean(x > 0 for x in deltas[entry["proposed"]]),
                                        control_improvement_fraction=statistics.mean(x > 0 for x in deltas[entry["control"]])))
        write_csv(output / "diagnostic_children.csv", child_rows)
        write_csv(output / "diagnostic_parents.csv", parent_rows)
        summary = []
        for d in datasets:
            for kind in ("search_best", "fixed_end"):
                rows = [r for r in parent_rows if r["dataset"] == d and r["protocol"] == kind]
                diffs = [r["paired_delta_pp"] for r in rows]
                distinct = [r["paired_delta_pp"] for r in rows if not r["same_sample"]]
                summary.append(dict(dataset=d, protocol=kind, paired_parents=len(rows),
                                    identical=sum(r["identical"] for r in rows),
                                    distinct_sample_parents=len(distinct),
                                    mean_delta_pp_distinct_samples=statistics.mean(distinct) if distinct else None,
                                    mean_delta_pp=statistics.mean(diffs) if diffs else None,
                                    wins=sum(x > 1e-9 for x in diffs), losses=sum(x < -1e-9 for x in diffs),
                                    ties=sum(abs(x) <= 1e-9 for x in diffs)))
        write_csv(output / "diagnostic_summary.csv", summary)


def under_cap(records, cap):
    eligible = [r for r in records if r["status"] == "complete" and r["macs"] <= cap]
    return len(eligible), min(eligible, key=lambda r: (r["objectives"][0], r["macs"], r["hash"]), default=None)


def _search_pair(cfg, dataset, seed, output, with_refit):
    from .cli import run_with_reuse
    from .data import DataProtocol
    from .followup import fit
    from .search import load_records, run_path, search

    source_variant = variant_for(cfg, dataset)
    synthetic = cfg["profile"] == "smoke"
    cap = cfg["temporal_study"]["datasets"][dataset]["cap"]
    initial = None
    summary, curves, refits, telemetry, edits = [], [], [], [], []
    for label, enabled in (("original", False), ("temporal", True)):
        arm = deepcopy(cfg)
        arm["search"]["temporal_mutation"]["enabled"] = enabled
        arm["output"] = str(output)
        if synthetic:
            arm["logging"]["tensorboard"] = False
        variant = "tm_" + label
        arm["experiments"] = {variant: arm["experiments"][source_variant]}
        path = run_path(arm, dataset, variant, seed, "main")
        run_with_reuse(path, "search", lambda resume: search(
            arm, dataset, variant, seed, "main", resume, synthetic))
        records = load_records(path)
        population = [r["hash"] for r in records if r["generation"] == 0]
        if initial is not None and population != initial:
            raise ValueError(f"Initial candidate batches differ: {dataset}/{seed}")
        initial = population
        state = read_json(path / "search_state.json")
        group = dict(dataset=dataset, search_seed=seed, arm=label)
        count, selected = under_cap(records, cap)
        good = [r for r in records if r["status"] == "complete"]
        row = dict(group, cap=cap, cap_hits=count, hit=selected is not None,
                   min_macs=min((r["macs"] for r in good), default=None),
                   hv=state["history"][-1]["hv"], elapsed_seconds=state["elapsed_seconds"],
                   initial_hashes=population)
        curves.extend(dict(group, **point) for point in state["history"])
        stats = state.get("temporal_mutation", {})
        telemetry.extend(dict(group, operator=mode, **counts) for mode, counts in stats.get("operators", {}).items())
        edits.extend(dict(group, child=r["hash"], child_macs=r["macs"], status=r["status"], **r["temporal_edit"])
                     for r in records if "temporal_edit" in r)
        if selected is not None:
            row.update(hash=selected["hash"], macs=selected["macs"], validation_f1=selected["validation"]["macro_f1"])
            if with_refit:
                manifest = read_json(path / "manifest.json")
                data = DataProtocol(arm, dataset, synthetic)
                if data.manifest != manifest["data"]:
                    raise ValueError("TRAIN changed after search")
                refit_dir = output / "refits" / dataset / variant / str(seed) / str(cap)
                rows = list(fit(refit_dir, arm, data, manifest["identity"],
                                {"under_cap": Architecture.from_dict(selected["architecture"])}))
                refits.extend({**r, **group} for r in rows)
                row["test_f1"] = statistics.mean(r["macro_f1"] for r in rows)
                latency = [r["batch1_median_ms"] for r in rows if r.get("batch1_median_ms") is not None]
                row["latency_batch1_median_ms"] = statistics.mean(latency) if latency else None
        summary.append(row)
    return summary, curves, refits, telemetry, edits


def paired_search(args, cfg, datasets, output):
    seeds = args.seed or cfg["seeds"]["search"]
    configs = cfg["temporal_study"]["datasets"]
    plan = {"datasets": {d: configs[d] for d in datasets}, "seeds": seeds, "population": cfg["search"]["population"],
            "budget": cfg["search"]["budget"], "searches": 2 * len(datasets) * len(seeds),
            "candidates": 2 * len(datasets) * len(seeds) * cfg["search"]["budget"],
            "with_refit": args.with_refit, "output": str(output), "dry_run": not args.execute}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    with run_lock(output / ".study.lock"):
        bind(output / "search_protocol.json", {"config": cfg, "implementation": study_protocol()})
        summary, curves, refits, telemetry, edits, pairs = [], [], [], [], [], []
        for d in datasets:
            for seed in seeds:
                result_path = output / "pairs" / d / f"{seed}.json"
                completed = read_json(result_path) if result_path.exists() else None
                if completed and completed["with_refit"] and not args.with_refit:
                    continue
                result = _search_pair(cfg, d, seed, output, args.with_refit)
                atomic_json(result_path, {"with_refit": args.with_refit, "results": result})
                _export_pairs(output)
        _export_pairs(output)


def _export_pairs(output):
    summary, curves, refits, telemetry, edits, pairs = [], [], [], [], [], []
    for path in sorted((output / "pairs").glob("*/*.json")):
        s, c, r, t, e = read_json(path)["results"]
        summary.extend(s); curves.extend(c); refits.extend(r); telemetry.extend(t); edits.extend(e)
        old, new = s

        def gap(key, mult=1):
            return mult * (new[key] - old[key]) if old.get(key) is not None and new.get(key) is not None else None

        pairs.append(dict(dataset=old["dataset"], search_seed=old["search_seed"],
                          old_hit=old["hit"], new_hit=new["hit"], hv_delta=gap("hv"),
                          cap_hits_delta=gap("cap_hits"), validation_f1_delta_pp=gap("validation_f1", 100),
                          test_f1_delta_pp=gap("test_f1", 100), latency_delta_ms=gap("latency_batch1_median_ms")))
    hit_summary = []
    for d, a in sorted({(r["dataset"], r["arm"]) for r in summary}):
        group = [r for r in summary if r["dataset"] == d and r["arm"] == a]
        hits = sum(r["hit"] for r in group)
        hit_summary.append(dict(dataset=d, arm=a, searches=len(group), searches_hit=hits, hit_rate=hits / len(group)))
    for name, rows in (("search_runs", summary), ("search_hv", curves), ("search_refits", refits),
                       ("search_paired", pairs), ("search_hit_summary", hit_summary),
                       ("mutation_counts", telemetry), ("mutation_edits", edits)):
        write_csv(output / f"{name}.csv", rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("diagnose", "search"):
        s = sub.add_parser(name)
        s.add_argument("--execute", action="store_true")
        s.add_argument("--profile", choices=("paper", "pilot", "smoke"), default="paper")
        s.add_argument("--dataset", action="append", choices=("Heartbeat", "PEMS-SF", "SpokenArabicDigits"))
        s.add_argument("--source-root", type=Path, default=Path("mamba_tree_nas/runs"))
        s.add_argument("--source-run-id", default="main")
        s.add_argument("--data-root", type=Path)
        s.add_argument("--output", type=Path)
        s.add_argument("--config", type=Path)
        s.add_argument("--set", dest="overrides", action="append", default=[])
        if name == "search":
            s.add_argument("--seed", type=int, action="append")
            s.add_argument("--with-refit", action="store_true")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.source_run_id) or args.source_run_id in (".", ".."):
        p.error("--source-run-id must be a simple directory name")
    datasets = list(dict.fromkeys(args.dataset or ["Heartbeat", "PEMS-SF"]))
    cfg = load_config(args.config, args.profile, args.overrides, multiscale=True, temporal=True)
    if args.data_root:
        cfg["data"]["root"] = str(args.data_root)
    output = resolve_path(args.output or Path("mamba_tree_nas/work/temporal_mutation") / args.mode / args.profile).resolve()
    source = resolve_path(args.source_root).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        p.error("Output must be separate from the original run tree")
    if args.mode == "search":
        if args.seed and len(set(args.seed)) != len(args.seed):
            p.error("--seed must be unique")
        paired_search(args, cfg, datasets, output)
    else:
        if args.mode == "diagnose" and getattr(args, "profile") == "pilot":
            p.error("Diagnose uses the saved source protocol or the synthetic smoke profile")
        diagnose(args, cfg, datasets, output)


if __name__ == "__main__":
    main()
