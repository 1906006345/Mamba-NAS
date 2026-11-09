"""Fixed-rate diagnostics and paired, cost-constrained multiscale search."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics

from .architecture import Architecture, Space, at, paths, substitute
from .artifacts import atomic_json, init_run, read_json, run_lock, write_csv
from .config import load_config
from .cost import macs


def leaves(arch):
    return [(p, at(arch.tree, p).leaf) for p in paths(arch.tree) if at(arch.tree, p).op == "OP"]


def with_rates(arch, rates):
    tree = arch.tree
    for (p, leaf), rate in zip(leaves(arch), rates, strict=True):
        tree = substitute(tree, p, replace(at(tree, p), leaf=replace(leaf, rate=rate)))
    return replace(arch, tree=tree)


def ordered_dict(arch):
    # Keep branch initialization order fixed even when a rate change flips PAR's hash sort.
    def encode(node):
        if node.op == "OP":
            return node.to_dict(arch.version)
        return {"op": node.op, "children": [encode(c) for c in node.children]}

    return dict(arch.to_dict(), tree=encode(arch.tree))


def source(root, run_id, dataset, direction, seed, role):
    path = root / dataset / ("tree_multiscale_" + direction) / f"seed_{seed}" / run_id
    manifest = read_json(path / "manifest.json")
    if read_json(path / "search_state.json")["status"] != "complete":
        raise ValueError(f"Incomplete source search: {path}")
    reps = read_json(path / "representatives.json")
    return manifest, Architecture.from_dict(reps["architectures"][reps["roles"][role]])


def fixed_plan(root, run_id):
    def load(dataset, direction, seed, role="high_f1"):
        return source(root, run_id, dataset, direction, seed, role)

    for dataset, direction, expected in (("Heartbeat", "forward", 4), ("PEMS-SF", "direction", 2)):
        m, a = load(dataset, direction, 2021)
        if [x.rate for _, x in leaves(a)] != [expected]:
            raise ValueError(f"Unexpected single-leaf representative for {dataset}")
        # PEMS-SF patch8 has 18 tokens; rate 4 would leave only 5 (<8).
        yield dataset + "_single", m, a, [(1,), (2,), (4,)] if expected == 4 else [(1,), (2,)]
    m, a = load("Heartbeat", "forward", 2023)
    if a.tree.op != "PAR" or sorted(x.rate for _, x in leaves(a)) != [2, 4]:
        raise ValueError("Expected the recorded two-leaf mixed Heartbeat tree")
    original = tuple(x.rate for _, x in leaves(a))
    yield "Heartbeat_parallel", m, a, [original, original[::-1], (2, 2), (4, 4), (1, 1)]
    for seed in (2021, 2022, 2023):
        m, a = load("SpokenArabicDigits", "direction", seed, "knee")
        original = tuple(x.rate for _, x in leaves(a))
        if max(original) > 1:
            yield "SpokenArabicDigits_knee", m, a, [original, (1,) * len(original)]
            break


def fit(output, cfg, data, identity, architectures):
    """Use the standard refit pipeline with a separate, immutable input manifest."""
    from .cli import run_with_reuse
    from .refit import refit

    archs = {a.hash: ordered_dict(a) for a in architectures.values()}
    roles = {label: a.hash for label, a in architectures.items()}
    identity = dict(identity, followup=roles)
    with run_lock(output / ".setup.lock"):
        init_run(output, cfg, identity, data.manifest, (output / "manifest.json").exists())
        atomic_json(output / "representatives.json", {"roles": roles, "architectures": archs})
        atomic_json(output / "search_state.json", {"status": "complete", "purpose": "fixed architecture refit input"})
    run_with_reuse(output, "refit", lambda resume: refit(output, resume=resume))
    for label, arch in architectures.items():
        cost = macs(arch, data.max_length, data.channels, len(data.classes))
        for seed in cfg["seeds"]["refit"]:
            r = read_json(output / "refit" / arch.hash / f"seed_{seed}.json")
            yield {"arm": label, "hash": arch.hash, "seed": seed, "macs": cost,
                   "parameters": r["parameters"], "epochs": r["epochs"],
                   "macro_f1": r["test"]["macro_f1"], **r["profile"]}


def fixed(args, output):
    from .data import DataProtocol

    cases = list(fixed_plan(args.source_root, args.source_run_id))
    plan, rows = [], []
    for label, manifest, arch, rates in cases:
        arms = {"rate_" + "_".join(map(str, r)): with_rates(arch, r) for r in rates}
        plan.append({"case": label, "source": manifest["identity"],
                     "arms": {k: ordered_dict(a) for k, a in arms.items()}})
    skipped = [] if any(c["case"] == "SpokenArabicDigits_knee" for c in plan) else [
        "SpokenArabicDigits_knee: all three representatives have rate=1"]
    print(json.dumps({"cases": plan, "skipped": skipped, "output": str(output),
                      "dry_run": not args.execute}, indent=2))
    if not args.execute:
        return
    prepared = []
    for (label, manifest, _, _), case in zip(cases, plan, strict=True):
        cfg, identity = manifest["config"], manifest["identity"]
        data = DataProtocol(cfg, identity["dataset"], identity["synthetic"])
        if data.manifest != manifest["data"]:
            raise ValueError(f"Source data or split changed: {label}")
        space = Space(cfg["space"], cfg["experiments"][identity["variant"]],
                      min(len(data.samples[i]) for i in data.train_indices))
        arms = {k: Architecture.from_dict(v) for k, v in case["arms"].items()}
        if not all(map(space.valid, arms.values())):
            raise ValueError(f"Illegal rate arm: {label}")
        prepared.append((label, cfg, data, identity, arms))
    atomic_json(output / "fixed_plan.json", {"cases": plan, "skipped": skipped})
    for label, cfg, data, identity, arms in prepared:
        for row in fit(output / label, cfg, data, identity, arms):
            rows.append(dict(row, case=label))
        write_csv(output / "fixed_results.csv", rows)
        paired = []
        for r in rows:
            base = next(b for b in rows if b["case"] == r["case"] and b["seed"] == r["seed"]
                        and all(v == "1" for v in b["arm"].split("_")[1:]))
            paired.append(dict(r, f1_delta_pp=100 * (r["macro_f1"] - base["macro_f1"]),
                               macs_delta=r["macs"] - base["macs"]))
        write_csv(output / "fixed_paired.csv", paired)


def budgeted(args, output):
    from .cli import run_with_reuse
    from .data import DataProtocol
    from .search import load_records, run_path, search

    cfg = load_config(profile=args.profile, multiscale=True)
    variant = "tree_multiscale_direction"
    synthetic = args.profile == "smoke"
    dataset = "SyntheticBudget" if synthetic else "SpokenArabicDigits"
    cfg["search"].update(population=3 if synthetic else 24, budget=5 if synthetic else 96)
    cap = args.cap if args.cap is not None else (10**9 if synthetic else 180_000)
    seeds = args.seed or cfg["seeds"]["search"]
    print(json.dumps({"dataset": dataset, "seeds": seeds, "rates": [[1], [1, 2, 4]],
                      "cap": cap, "evaluations_upper": 2 * len(seeds) * cfg["search"]["budget"],
                      "output": str(output), "dry_run": not args.execute}, indent=2))
    if not args.execute:
        return
    selections, fits = [], []
    for label, rates in (("rate1", [1]), ("rates124", [1, 2, 4])):
        arm = load_config(profile=args.profile, multiscale=True)
        arm["search"] = cfg["search"]
        arm["experiments"][variant]["multiscale"]["rates"] = rates
        arm["output"] = str(output / label)
        if synthetic:
            arm["logging"]["tensorboard"] = False
        for seed in seeds:
            path = run_path(arm, dataset, variant, seed, "search")
            run_with_reuse(path, "search", lambda resume: search(
                arm, dataset, variant, seed, "search", resume, synthetic))
            records = [r for r in load_records(path) if r["status"] == "complete"]
            eligible = [r for r in records if r["macs"] <= cap]
            state = read_json(path / "search_state.json")
            row = {"arm": label, "search_seed": seed, "cap": cap,
                   "min_macs": min((r["macs"] for r in records), default=None),
                   "hv": state["history"][-1]["hv"], "eligible": len(eligible)}
            if eligible:
                chosen = min(eligible, key=lambda r: (r["objectives"][0], r["macs"], r["hash"]))
                manifest = read_json(path / "manifest.json")
                data = DataProtocol(arm, dataset, synthetic)
                if data.manifest != manifest["data"]:
                    raise ValueError("TRAIN changed after search")
                results = list(fit(output / label / f"seed_{seed}_cap_{cap}", arm, data,
                                   manifest["identity"], {"under_cap": Architecture.from_dict(chosen["architecture"])}))
                fits.extend(dict(r, arm=label, search_seed=seed) for r in results)
                row.update(hash=chosen["hash"], macs=chosen["macs"], validation_f1=chosen["validation"]["macro_f1"],
                           test_f1=statistics.mean(r["macro_f1"] for r in results))
            selections.append(row)
            write_csv(output / f"budget_{cap}_search.csv", selections)
            write_csv(output / f"budget_{cap}_refit.csv", fits)
    pairs = []
    for seed in seeds:
        a, b = (next(r for r in selections if r["arm"] == label and r["search_seed"] == seed)
                for label in ("rate1", "rates124"))
        pairs.append({"search_seed": seed, "rate1_hit": bool(a["eligible"]),
                      "rates124_hit": bool(b["eligible"]),
                      "f1_delta_pp": 100 * (b["test_f1"] - a["test_f1"]) if "test_f1" in a and "test_f1" in b else None})
    write_csv(output / f"budget_{cap}_paired.csv", pairs)
    summary = []
    for label in ("rate1", "rates124"):
        group = [r for r in selections if r["arm"] == label]
        f1 = [r["test_f1"] for r in group if "test_f1" in r]
        summary.append({"arm": label, "seeds": len(group), "cap_hits": len(f1),
                        "test_f1_mean_on_hits": statistics.mean(f1) if f1 else None,
                        "test_f1_sd_on_hits": statistics.stdev(f1) if len(f1) > 1 else None})
    write_csv(output / f"budget_{cap}_summary.csv", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fixed", "budgeted"))
    p.add_argument("--execute", action="store_true", help="Without this flag only print the plan")
    p.add_argument("--output", type=Path)
    p.add_argument("--source-root", type=Path, default=Path("mamba_tree_nas/runs"))
    p.add_argument("--source-run-id", default="main")
    p.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    p.add_argument("--seed", type=int, action="append")
    p.add_argument("--cap", type=int)
    args = p.parse_args()
    if Path(args.source_run_id).name != args.source_run_id or args.source_run_id in ("", ".", ".."):
        p.error("--source-run-id must be a simple directory name")
    output = (args.output or Path("mamba_tree_nas/work/followup") / args.mode / args.profile).resolve()
    root = args.source_root.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        p.error("Output must be separate from the original run tree")
    if args.cap is not None and args.cap < 1:
        p.error("--cap must be positive")
    if args.seed and len(set(args.seed)) != len(args.seed):
        p.error("--seed must be unique")
    (fixed if args.mode == "fixed" else budgeted)(args, output)


if __name__ == "__main__":
    main()
