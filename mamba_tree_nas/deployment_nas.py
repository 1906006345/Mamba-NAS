"""Opt-in deployment objectives, tree rank diagnosis and one-shot comparison."""
import argparse
import json
from pathlib import Path

from .artifacts import run_lock
from .config import resolve_path
from .deployment_protocol import Study, bind_study, configuration, runtime_identity


def preview(cfg, stage, datasets, seeds, output):
    s = cfg["deployment_nas"]
    runs = len(datasets) * len(seeds)
    groups = 3 if stage == "objectives" else 2 if stage == "compare" else 0
    return {
        "stage": stage, "profile": cfg["profile"], "datasets": datasets, "search_seeds": list(seeds),
        "output": str(output), "short_epochs_max": cfg["train"]["epochs"],
        "reference_epochs": s["reference_epochs"], "supernet_epochs": s["supernet_epochs"] if stage == "compare" else None,
        "panel_per_dataset": s["panel_size"], "search_runs_upper": runs * groups,
        "logical_candidates_upper": runs * groups * s["search"]["budget"],
        "rank_fits_upper": len(datasets) * s["panel_size"] * len(s["training_seeds"]) * 2 if stage == "rank" else 0,
        "reference_audit_fits_upper": (
            runs * groups * len(s["checkpoints"]) * s["audit_quota"] * len(s["training_seeds"])
            if stage == "compare" else 0),
        "real_hardware": cfg["profile"] != "smoke",
    }


def _check_initial(states, count):
    initial = [[row["hash"] for row in state["records"][:count]] for state in states]
    if any(len(hashes) != count for hashes in initial) or any(hashes != initial[0] for hashes in initial):
        raise RuntimeError("Paired methods used different initial populations")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "objectives", "rank", "compare", "status", "report"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "paper"), default="paper")
    parser.add_argument("--dataset", action="append", help="Repeat to select multiple configured datasets")
    parser.add_argument("--seed", type=int, action="append", help="Repeat to select configured search seeds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--execute", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = resolve_path(args.output or Path("mamba_tree_nas/work/deployment_nas") / args.profile).resolve()
    if args.stage in ("status", "report"):
        from .deployment_report import collect, write_report

        report, online, audited = collect(output)
        if args.stage == "report" and args.execute and report["status"] != "not_started":
            with run_lock(output / ".deployment.lock"):
                report, online, audited = collect(output)
                write_report(output, report, online, audited)
        if args.dataset:
            report["datasets"] = [r for r in report["datasets"] if r["dataset"] in args.dataset]
            report["runs"] = [r for r in report["runs"] if r["dataset"] in args.dataset]
        if args.seed:
            report["runs"] = [r for r in report["runs"] if r["seed"] in args.seed]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    cfg = configuration(args.profile, args.config, args.data_root)
    s = cfg["deployment_nas"]
    datasets = list(dict.fromkeys(args.dataset or (["Heartbeat"] if args.profile != "paper" else s["datasets"])))
    seeds = list(dict.fromkeys(args.seed or s["search_seeds"]))
    if set(datasets) - set(s["datasets"]) or set(seeds) - set(s["search_seeds"]):
        parser.error("Select datasets and seeds present in the deployment configuration")
    if args.seed and args.stage not in ("compare", "objectives"):
        parser.error("--seed is only a subset selector for the search stages")
    print(json.dumps(dict(preview(cfg, args.stage, datasets, seeds, output), execute=args.execute), indent=2), flush=True)
    if not args.execute:
        return
    if output.exists() and not (output / "protocol.json").exists() and any(
            p.name != ".deployment.lock" for p in output.iterdir()):
        raise ValueError("Use an empty output folder or an existing deployment study")
    runtime = runtime_identity(cfg)
    failed_rank, failed_audits = False, False
    with run_lock(output / ".deployment.lock"):
        fingerprint = bind_study(output, cfg, runtime)
        for dataset in datasets:
            study = Study(cfg, output, dataset, fingerprint)
            if args.stage == "benchmark":
                from .deployment_benchmark import benchmark

                benchmark(study)
            elif args.stage == "rank":
                from .deployment_rank import rank

                failed_rank |= rank(study)["status"] != "complete"
            else:
                from .deployment_search import ARMS, SUPERNET, audit_front, run_search

                arms = ARMS if args.stage == "objectives" else (ARMS[2], SUPERNET)
                for seed in seeds:
                    states = [run_search(study, arm, seed) for arm in arms]
                    _check_initial(states, cfg["search"]["population"])
                    if args.stage == "compare":
                        for arm, state in zip(arms, states):
                            failed_audits |= audit_front(study, arm, seed, state)[-1]["failed_audits"] > 0
    if failed_rank or failed_audits:
        raise RuntimeError("Some ranking or audit fits failed. Results were saved; re-run to retry missing fits.")


if __name__ == "__main__":
    main()
