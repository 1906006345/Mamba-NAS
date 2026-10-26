from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from .artifacts import environment, read_json
from .config import load_config, resolve_path


def parser():
    p = argparse.ArgumentParser(description="Independent supervised Mamba functional-tree NAS")
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("verify-environment", "download", "smoke", "search", "refit", "run-suite", "status", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--config")
        sub.add_argument("--profile", choices=("smoke", "pilot", "paper"), default=None)
        sub.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
        sub.add_argument("--dataset")
        sub.add_argument("--variant")
        sub.add_argument("--search-seed", type=int)
        sub.add_argument("--run-id", default="main")
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--dry-run", action="store_true")
        sub.add_argument("--run", type=Path, help="Specific saved run; refit uses its resolved config")
        sub.add_argument("--root", type=Path, help="Run tree for status/report")
        sub.add_argument("--output", type=Path, help="Report export directory")
        sub.add_argument("--suite", choices=("primary", "uea10"), default="primary")
        sub.add_argument("--with-refit", action="store_true")
        sub.add_argument("--external-baselines", type=Path)
        sub.add_argument("--gpu-test", action="store_true", help="Real official Mamba forward/backward environment acceptance")
    return p


def matrix(cfg, args):
    datasets = [args.dataset] if args.dataset else cfg["data"][args.suite]
    variants = [args.variant] if args.variant else list(cfg["experiments"])
    seeds = [args.search_seed] if args.search_seed is not None else cfg["seeds"]["search"]
    for variant in variants:
        if variant not in cfg["experiments"]:
            raise ValueError(f"Unknown experiment: {variant}")
    return [(d, v, s) for d in datasets for v in variants for s in seeds]


def budget_summary(cfg, runs, run_id, with_refit):
    from .search import run_path
    n = len(runs)
    return {"profile": cfg["profile"], "backend": cfg["backend"], "search_runs": n, "unique_candidate_budget_per_search": cfg["search"]["budget"], "candidate_evaluations_upper": n * cfg["search"]["budget"], "proxy_initializations_upper": n * cfg["search"]["budget"] * len(cfg["seeds"]["proxy"]) if cfg["proxy"]["enabled"] else 0, "calibrations_upper_if_refit": n * 3, "refit_fits_upper_if_refit": n * 3 * len(cfg["seeds"]["refit"]), "will_refit": with_refit, "deduplication": "role aliases reduce calibration/refit counts", "output_root": str(resolve_path(cfg["output"])), "runs": [{"dataset": d, "variant": v, "search_seed": s, "path": str(run_path(cfg, d, v, s, run_id))} for d, v, s in runs]}


def download(cfg, datasets, dry_run=False):
    import shutil
    import tempfile
    import urllib.request
    import zipfile
    plan = []
    for dataset in datasets:
        destination = resolve_path(cfg["data"]["root"]) / dataset
        url = cfg["data"]["download_url"].format(dataset=dataset)
        plan.append({"dataset": dataset, "url": url, "destination": str(destination)})
        if dry_run or all((destination / f"{dataset}_{split}.ts").exists() for split in ("TRAIN", "TEST")):
            continue
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination) as temp:
            archive = Path(temp) / "dataset.zip"
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive) as z:
                # Extract only required .ts sources, flatten safely and never unpack
                # arbitrary archive paths or unrelated large files.
                for split in ("TRAIN", "TEST"):
                    expected = f"{dataset}_{split}.ts"
                    matches = [x for x in z.namelist() if Path(x).name.lower() == expected.lower()]
                    if len(matches) != 1:
                        raise ValueError(f"Archive must contain exactly one {expected}")
                    staged = Path(temp) / expected
                    with z.open(matches[0]) as src, staged.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if not (destination / expected).exists():
                        staged.replace(destination / expected)
    return plan


def verify(cfg, gpu_test=False):
    result = environment()
    result["production_prerequisites_present"] = bool(result["cuda_available"] and sys.platform.startswith("linux") and result["packages"]["mamba-ssm"])
    result["production_ready"] = False
    result["gpu_acceptance"] = "not_run"
    if gpu_test:
        import torch
        from .model import mixer_factory
        with torch.enable_grad():
            m = mixer_factory("mamba", cfg["device"])(d_model=16, d_state=8, d_conv=4, expand=2).to(cfg["device"])
            x = torch.randn(2, 17, 16, device=cfg["device"], requires_grad=True)
            y = m(x); y.square().mean().backward()
            if not torch.isfinite(y).all() or x.grad is None or not torch.isfinite(x.grad).all():
                raise RuntimeError("Real Mamba GPU forward/backward failed")
            result["gpu_acceptance"] = "passed_real_mamba_forward_backward"
            result["production_ready"] = True
    return result


def status(root):
    root = Path(root)
    files = [root / "search_state.json"] if (root / "search_state.json").exists() else sorted(root.rglob("search_state.json"))
    results = []
    for file in files:
        state = read_json(file)
        manifest = read_json(file.parent / "manifest.json")
        from .search import load_records
        records = load_records(file.parent)
        good = [r for r in records if r["status"] == "complete"]
        refit_path = file.parent / "refit" / "state.json"
        results.append({"path": str(file.parent), **manifest["identity"], "status": state["status"], "completed": len(state["completed"]), "budget": manifest["config"]["search"]["budget"], "generation": state["generation"], "current_candidate": state.get("current_candidate"), "current_epoch": state.get("current_epoch"), "best_validation_macro_f1": max((r["validation"]["macro_f1"] for r in good), default=None), "elapsed_seconds": state["elapsed_seconds"], "failures": [r["error"]["message"] for r in records if r["status"] == "failed"], "last_error": state.get("last_error"), "refit": read_json(refit_path) if refit_path.exists() else "not_started"})
    return results


def main(argv=None):
    args = parser().parse_args(argv)
    profile = args.profile or ("smoke" if args.command == "smoke" else "paper")
    cfg = load_config(args.config, profile, args.overrides)
    if args.command == "verify-environment":
        result = verify(cfg, args.gpu_test and not args.dry_run)
    elif args.command == "download":
        result = download(cfg, [args.dataset] if args.dataset else cfg["data"][args.suite], args.dry_run)
    elif args.command in ("search", "run-suite", "smoke"):
        from .search import run_path, search
        if args.command == "smoke":
            if cfg["backend"] != "test" or cfg["device"] != "cpu":
                raise ValueError("smoke requires CPU test backend; real GPU acceptance uses search --profile pilot")
            runs = [(args.dataset or "SyntheticSmoke", args.variant or "tree_direction", args.search_seed if args.search_seed is not None else cfg["seeds"]["search"][0])]
        elif args.command == "search":
            if not args.dataset or not args.variant:
                raise ValueError("search requires --dataset and --variant")
            runs = [(args.dataset, args.variant, args.search_seed if args.search_seed is not None else cfg["seeds"]["search"][0])]
        else:
            runs = matrix(cfg, args)
        result = budget_summary(cfg, runs, args.run_id, args.with_refit or args.command == "smoke")
        if not args.dry_run:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            for dataset, variant, seed in runs:
                planned_path = run_path(cfg, dataset, variant, seed, args.run_id)
                resume_search = args.resume and ((planned_path / "manifest.json").exists() if args.command == "run-suite" else True)
                path = search(cfg, dataset, variant, seed, args.run_id, resume_search, synthetic=args.command == "smoke")
                if args.with_refit or args.command == "smoke":
                    from .refit import refit
                    refit(path, resume=args.resume and (path / "refit" / "state.json").exists())
                if args.command == "smoke":
                    from .report import report
                    report(path, args.output or resolve_path(cfg["exports"]) / "smoke")
            result["completed"] = True
    elif args.command == "refit":
        from .search import run_path
        from .refit import refit
        if not args.run and (not args.dataset or not args.variant):
            raise ValueError("refit requires --run or --dataset and --variant")
        path = args.run or run_path(cfg, args.dataset, args.variant, args.search_seed if args.search_seed is not None else cfg["seeds"]["search"][0], args.run_id)
        manifest = read_json(path / "manifest.json")
        if (args.config or args.profile or args.overrides) and cfg != manifest["config"]:
            raise ValueError("Explicit refit configuration conflicts with search manifest")
        result = {"run": str(path), "config_source": "saved search manifest", "refit_fits_upper": 3 * len(manifest["config"]["seeds"]["refit"])}
        if not args.dry_run:
            result["output"] = str(refit(path, resume=args.resume))
    elif args.command == "status":
        result = status(args.run or args.root or resolve_path(cfg["output"]))
    elif args.command == "report":
        from .report import report
        root = args.run or args.root or resolve_path(cfg["output"])
        output = args.output or resolve_path(cfg["exports"])
        result = {"input": str(root), "output": str(output)}
        if not args.dry_run:
            result["output"] = str(report(root, output, args.external_baselines))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
