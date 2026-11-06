"""Dedicated launcher and read-only comparisons for the multiscale tree."""
from pathlib import Path
import json
import re

from .artifacts import atomic_json, read_json, write_csv
from .config import resolve_path


BASELINES = {
    "tree_multiscale_forward": "tree_forward",
    "tree_multiscale_direction": "tree_direction",
}


def comparison_row(new, old):
    row = {"new_path": str(new), "baseline_path": str(old)}
    missing = [label for label, path in (("new", new), ("baseline", old))
               if not all((path / f).is_file() for f in ("manifest.json", "search_state.json"))]
    if missing:
        return dict(row, status="missing", reasons=missing)
    a, b = (read_json(p / "manifest.json") for p in (new, old))
    x, y = (read_json(p / "search_state.json") for p in (new, old))
    if any(s["status"] != "complete" or not s["history"] for s in (x, y)):
        return dict(row, status="incomplete", reasons=[])
    ca, cb = a["config"], b["config"]
    reasons = [k for k in ("data",) if a[k] != b[k]]
    ia, ib = a["identity"], b["identity"]
    reasons += ["identity." + k for k in ("dataset", "search_seed", "synthetic") if ia[k] != ib[k]]
    if BASELINES.get(ia["variant"]) != ib["variant"]:
        reasons.append("baseline_variant")
    reasons += [k for k in ("backend", "runtime", "train", "space", "proxy", "seeds")
                if ca[k] != cb[k]]
    reasons += ["search." + k for k in sorted(set(ca["search"]) | set(cb["search"]))
                if ca["search"].get(k) != cb["search"].get(k)]
    if x["cost_bound"] != y["cost_bound"]:
        reasons.append("cost_bound")
    if reasons:
        return dict(row, status="protocol_mismatch", reasons=reasons)
    new_hv, old_hv = x["history"][-1]["hv"], y["history"][-1]["hv"]
    return dict(row, status="comparable", reasons=[], multiscale_hv=new_hv,
                baseline_hv=old_hv, hv_delta=new_hv - old_hv)


def gpu_check(device):
    import torch
    from .architecture import Leaf, Node
    from .model import TreeModule, mixer_factory

    factory = mixer_factory("mamba", device)
    for rate in (2, 4):
        for direction in ("F", "B", "FB_shared"):
            tree = TreeModule(Node("OP", leaf=Leaf(direction, 8, 4, 1, rate)), 64, 0, factory).to(device)
            x = torch.randn(2, 35, 64, device=device, requires_grad=True)
            mask = torch.arange(35, device=device)[None, :] < torch.tensor([[35], [31]], device=device)
            y = tree(x, mask)
            y.square().mean().backward()
            if y.shape != x.shape or not torch.isfinite(y).all() or x.grad is None or not torch.isfinite(x.grad).all():
                raise RuntimeError(f"Multiscale Mamba GPU check failed: {rate=}, {direction=}")


def launch(cfg, args):
    from .cli import budget_summary, existing_run_action, matrix, run_with_reuse
    from .search import run_path, search

    synthetic = cfg["profile"] == "smoke"
    args.dataset = args.dataset or ("SyntheticMultiscale" if synthetic else "Heartbeat")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.dataset):
        raise ValueError("Dataset must be a simple directory name")
    if set(cfg["experiments"]) != set(BASELINES):
        raise ValueError("The multiscale launcher only accepts its two dedicated variants")
    runs = matrix(cfg, args)
    with_refit = (args.with_refit or synthetic) and not args.report_only
    result = budget_summary(cfg, runs, args.run_id, with_refit)
    output = Path(args.output or resolve_path(cfg["exports"]) / "multiscale" / args.run_id).resolve()
    base_cfg = {**cfg, "output": args.baseline_root or cfg["output"]}
    pairs = [(run_path(cfg, d, v, s, args.run_id),
              run_path(base_cfg, d, BASELINES[v], s, args.baseline_run_id)) for d, v, s in runs]
    protected_roots = [resolve_path(c["output"]).resolve() for c in (cfg, base_cfg)]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in protected_roots):
        raise ValueError("Report output must be separate from both run trees")
    result.update(report_only=args.report_only, report_output=str(output), run_actions=[
        {"path": str(new), "baseline_path": str(old),
         "search": "read_only" if args.report_only else existing_run_action(new, "search", args.existing),
         **({"refit": existing_run_action(new, "refit", args.existing)} if with_refit else {})}
        for new, old in pairs])
    if args.dry_run:
        return result
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not args.report_only:
        if args.multiscale_gpu_check:
            gpu_check(cfg["device"])
        for (dataset, variant, seed), (path, _) in zip(runs, pairs):
            run_with_reuse(path, "search", lambda resume: search(
                cfg, dataset, variant, seed, args.run_id, resume, synthetic), args.existing)
            if with_refit:
                from .refit import refit
                run_with_reuse(path, "refit", lambda resume: refit(path, resume=resume), args.existing)
    comparisons = [comparison_row(*pair) for pair in pairs]
    atomic_json(output / "baseline_comparison.json", comparisons)
    write_csv(output / "baseline_comparison.csv", comparisons)
    available = [p for p, _ in pairs if (p / "manifest.json").exists() and (p / "search_state.json").exists()]
    if available:
        from .report import report
        report(resolve_path(cfg["output"]), output, args.external_baselines, run_paths=available)
    return dict(result, completed=True)
