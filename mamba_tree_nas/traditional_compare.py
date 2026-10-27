"""Launch and report traditional serial baselines without altering tree search."""
from __future__ import annotations

from pathlib import Path
import statistics

from .artifacts import atomic_json, read_json, write_csv
from .config import digest, resolve_path
from .report import hv_auc, report
from .search import load_records, run_path


def matrix(cfg, args):
    if args.variant:
        raise ValueError("Use --algorithm nsga2/random/both for traditional baselines; --variant belongs to tree experiments")
    synthetic = cfg["profile"] == "smoke"
    if synthetic and (cfg["backend"] != "test" or cfg["device"] != "cpu"):
        raise ValueError("smoke comparison requires the CPU test backend")
    datasets = [args.dataset] if args.dataset else (["SyntheticSmoke"] if synthetic else cfg["data"][args.suite])
    seeds = [args.search_seed] if args.search_seed is not None else cfg["seeds"]["search"]
    methods = ("nsga2", "random") if args.algorithm == "both" else (args.algorithm,)
    return [(d, cfg["traditional"]["variants"][a], s) for d in datasets for s in seeds for a in methods]


def comparison_report(cfg, runs, run_id, output, include_tree=False, tree_run_id=None):
    paths = [run_path(cfg, d, v, s, run_id) for d, v, s in runs]
    if include_tree:
        reference = cfg["traditional"]["reference_tree"]
        if cfg["experiments"].get(reference, {}).get("topology") != "tree":
            raise ValueError("traditional_baselines.reference_tree must name a functional-tree experiment")
        serial = cfg["experiments"][cfg["traditional"]["serial_experiment"]]
        if set(cfg["experiments"][reference]["directions"]) != set(serial["directions"]):
            raise ValueError("Reference tree and serial baselines must use the same scanning directions")
        paths.extend(run_path(cfg, d, reference, s, tree_run_id or run_id) for d, s in dict.fromkeys((d, s) for d, _, s in runs))
    # Reject unequal budgets/HV conventions before the shared report writes files.
    # Its own preflight additionally checks data, source and evaluation protocols.
    conventions, rows = {}, []
    for path in paths:
        manifest = read_json(path / "manifest.json")
        saved, identity = manifest["config"], manifest["identity"]
        state = read_json(path / "search_state.json")
        key = identity["dataset"]
        experiment = saved["experiments"][identity["variant"]]
        convention = digest({"budget": saved["search"]["budget"], "population": saved["search"]["population"], "hv_reference": saved["search"]["hv_reference"], "cost_bound": state["cost_bound"], "directions": sorted(experiment["directions"])})
        if key in conventions and conventions[key] != convention:
            raise ValueError("Comparison refuses unequal search budgets, population sizes, scan directions or HV normalization/reference conventions")
        conventions[key] = convention
        is_traditional = identity["variant"] in saved.get("traditional", {}).get("variants", {}).values()
        records = load_records(path)
        good = [r for r in records if r["status"] == "complete"]
        row = dict(identity, algorithm=experiment["algorithm"], encoding="categorical_serial" if is_traditional else "functional_tree", status=state["status"], budget=saved["search"]["budget"], unique_candidates=len(records), failures=len(records) - len(good), final_hv=state["history"][-1]["hv"] if state["history"] else 0.0, search_seconds=state["elapsed_seconds"], proxy_seconds=sum(r["proxy"]["seconds"] for r in records), training_seconds=sum(r.get("training_seconds", 0) for r in records), best_validation_macro_f1=max((r["validation"]["macro_f1"] for r in good), default=None))
        for axis in ("candidate_count", "elapsed_seconds"):
            row.update({axis + "_hv_" + k: v for k, v in hv_auc(state["history"], axis).items() if k != "integration"})
        if state["history"]:
            row.update({k: state["history"][-1][k] for k in ("pareto_size", "failure_rate", "topology_count", "topology_entropy")})
        rows.append(row)
    output = Path(output)
    report(resolve_path(cfg["output"]), output, run_paths=paths)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["variant"]), []).append(row)
    summaries = []
    for (dataset, variant), values in sorted(grouped.items()):
        complete = [r for r in values if r["status"] == "complete" and r["unique_candidates"] == r["budget"]]
        item = {"dataset": dataset, "variant": variant, "algorithm": values[0]["algorithm"], "encoding": values[0]["encoding"], "search_count": len(complete), "selected_search_count": len(values), "all_selected_searches_complete": len(complete) == len(values), "independent_search_seeds": [r["search_seed"] for r in complete], "statistics": "mean and sample std across complete independent searches; partial searches excluded"}
        for metric in ("final_hv", "search_seconds", "proxy_seconds", "training_seconds", "best_validation_macro_f1", "candidate_count_hv_auc", "candidate_count_hv_mean_hv", "elapsed_seconds_hv_auc", "elapsed_seconds_hv_mean_hv", "pareto_size", "failure_rate", "topology_count", "topology_entropy"):
            scores = [r[metric] for r in complete if r.get(metric) is not None]
            item[metric + "_mean"] = statistics.mean(scores) if scores else None
            item[metric + "_std"] = statistics.stdev(scores) if len(scores) > 1 else None
        summaries.append(item)
    write_csv(output / "search_comparison_runs.csv", rows)
    write_csv(output / "search_comparison.csv", summaries)
    atomic_json(output / "comparison_manifest.json", {"runs": [str(p) for p in paths], "include_tree": include_tree, "comparison": "traditional categorical serial NSGA-II / independent Random Search; optional unchanged functional-tree reference", "controlled": "same Mamba leaf options, width/tokenizer/pooling, maximum block count, split, evaluation, unique budget, fixed HV bounds and refit protocol", "interpretation": "serial baselines versus a functional tree compare representation/topology and search operators together; not an isolated search-algorithm ablation", "random_search": "sampling never uses measured objectives; Pareto filtering is for shared reporting/representative selection only", "metrics": "shared mamba_tree_nas.metrics, cost, training, proxy and refit; no mamba_nas imports", "synthetic": any(r["synthetic"] for r in rows)})
    return output
