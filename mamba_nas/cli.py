from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import SearchConfig
from .constants import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, UEA10
from .data import download_dataset
from .export import export_paper
from .refit import refit_run, resolve_run
from .search import new_run_id, run_search


LOGGER = logging.getLogger(__name__)


def _common_search(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--validation-strategy",
        choices=("holdout", "kfold"),
        default="holdout",
        help="Candidate evaluation protocol; holdout preserves the original fixed 80/20 split",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of shared stratified folds when --validation-strategy=kfold",
    )
    parser.add_argument(
        "--search-algorithm",
        choices=("random", "nsga2", "moead"),
        default="nsga2",
        help="Search engine: independent random sampling, NSGA-II, or MOEA/D",
    )
    parser.add_argument(
        "--search-variant",
        choices=("baseline", "dual_crowding", "diversity_archive", "dual_space", "local_search", "archive_local_search", "archive_local_search_v2", "full", "scan_order", "scan_order_local_search", "scan_order_coupled_local_search", "scan_order_dual_crowding"),
        default="baseline",
        help="Evolution variant; baseline preserves vanilla NSGA-II",
    )
    parser.add_argument(
        "--objective-weight",
        type=float,
        default=0.70,
        help="Objective-space crowding weight used by decision-crowding variants",
    )
    parser.add_argument(
        "--decision-weight",
        type=float,
        default=0.30,
        help="Decision-space crowding weight; both crowding weights must sum to 1",
    )
    parser.add_argument("--local-search-ratio", type=float, default=None)
    parser.add_argument("--coupled-neighbor-ratio", type=float, default=0.30)
    parser.add_argument(
        "--hypervolume-space", choices=("active", "scan_order"), default=None,
        help="Search space used to define the common Hypervolume reference point",
    )
    parser.add_argument("--archive-seed-ratio", type=float, default=None)
    parser.add_argument("--pareto-seed-ratio", type=float, default=None)
    parser.add_argument("--population-seed-ratio", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mamba-tsc-nas")
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="Download reproducible UEA source files")
    download.add_argument("--suite", choices=("uea10",), default="uea10")
    download.add_argument("--dataset", choices=UEA10)
    download.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    download.add_argument("--force", action="store_true")

    search = subcommands.add_parser("search", help="Search one UEA dataset")
    search.add_argument("--dataset", choices=UEA10, required=True)
    _common_search(search)

    search_all = subcommands.add_parser("search-all", help="Search all UEA10 datasets sequentially")
    search_all.add_argument("--suite", choices=("uea10",), default="uea10")
    _common_search(search_all)

    smoke = subcommands.add_parser("synthetic-smoke", help="Download-free CPU/GPU pipeline check")
    _common_search(smoke)

    refit = subcommands.add_parser("refit", help="Retrain Pareto representatives and access TEST")
    refit.add_argument("--run", required=True)
    refit.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    refit.add_argument("--no-profile", action="store_true")

    export = subcommands.add_parser("export-paper", help="Regenerate tracked paper tables from artifacts")
    export.add_argument("--run", required=True)
    export.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    export.add_argument("--export-dir", default="paper_exports")

    subcommands.add_parser("verify-environment", help="Check Linux/CUDA/official Mamba forward/backward")
    return parser


def _config(args) -> SearchConfig:
    revised = args.search_variant == "archive_local_search_v2"
    scan_local = args.search_variant in {
        "scan_order_local_search", "scan_order_coupled_local_search"
    }
    return SearchConfig.from_budget(
        args.budget,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        validation_strategy=args.validation_strategy,
        cv_folds=args.cv_folds,
        search_algorithm=args.search_algorithm,
        search_variant=args.search_variant,
        objective_weight=args.objective_weight,
        decision_weight=args.decision_weight,
        local_search_ratio=(0.20 if revised or scan_local else 0.30) if args.local_search_ratio is None else args.local_search_ratio,
        coupled_neighbor_ratio=args.coupled_neighbor_ratio,
        archive_seed_ratio=0.30 if args.archive_seed_ratio is None else args.archive_seed_ratio,
        pareto_seed_ratio=0.50 if args.pareto_seed_ratio is None else args.pareto_seed_ratio,
        population_seed_ratio=0.20 if args.population_seed_ratio is None else args.population_seed_ratio,
        use_decision_crowding=args.search_variant in {"dual_crowding", "dual_space", "full", "scan_order_dual_crowding"},
        use_diversity_archive=args.search_variant in {"diversity_archive", "dual_space", "archive_local_search", "archive_local_search_v2", "full"},
        use_pareto_local_search=args.search_variant in {
            "local_search", "archive_local_search", "archive_local_search_v2", "full",
            "scan_order_local_search", "scan_order_coupled_local_search",
        },
        use_macro_micro_archive=args.search_variant == "archive_local_search_v2",
        use_scan_order=args.search_variant in {
            "scan_order", "scan_order_local_search", "scan_order_coupled_local_search",
            "scan_order_dual_crowding",
        },
        use_coupled_local_search=args.search_variant == "scan_order_coupled_local_search",
        hypervolume_space=(
            "scan_order" if args.search_variant in {
                "scan_order", "scan_order_local_search",
            "scan_order_coupled_local_search", "scan_order_dual_crowding",
            } else "active"
        ) if args.hypervolume_space is None else args.hypervolume_space,
    ).validate()


def verify_environment() -> dict:
    import platform

    report = {"platform": platform.platform(), "linux": platform.system() == "Linux"}
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
        if not report["linux"]:
            raise RuntimeError("Official CUDA Mamba verification is supported only inside Linux/WSL2")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        from mamba_ssm import Mamba

        model = Mamba(d_model=64, d_state=8, d_conv=2, expand=1).cuda()
        values = torch.randn(2, 32, 64, device="cuda", requires_grad=True)
        output = model(values)
        output.square().mean().backward()
        report.update({"mamba_forward_backward": True, "output_shape": list(output.shape)})
    except Exception as exc:
        report.update({"mamba_forward_backward": False, "error": repr(exc)})
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "resume", False) and not getattr(args, "run_id", None):
        raise SystemExit("--resume requires --run-id so the saved generation can be located")
    if args.command == "download":
        datasets = [args.dataset] if args.dataset else UEA10
        for dataset in datasets:
            paths = download_dataset(args.data_dir, dataset, force=args.force)
            print(f"{dataset}: {', '.join(str(path) for path in paths)}")
        return 0
    if args.command == "search":
        path = run_search(args.dataset, _config(args), args.run_id, args.resume)
        LOGGER.info("[命令完成] search output=%s", path)
        print(path)
        return 0
    if args.command == "search-all":
        run_id = args.run_id or new_run_id("uea10")
        for dataset in UEA10:
            path = run_search(dataset, _config(args), run_id, args.resume)
            LOGGER.info("[数据集完成] dataset=%s output=%s", dataset, path)
            print(path)
        LOGGER.info("[命令完成] search-all run_id=%s", run_id)
        return 0
    if args.command == "synthetic-smoke":
        path = run_search("SyntheticUEA", _config(args), args.run_id, args.resume, synthetic=True)
        LOGGER.info("[命令完成] synthetic-smoke output=%s", path)
        print(path)
        return 0
    if args.command == "refit":
        run_dir = resolve_run(args.output_dir, args.run)
        refit_run(run_dir, profile=not args.no_profile)
        LOGGER.info("[命令完成] refit run=%s", run_dir)
        return 0
    if args.command == "export-paper":
        destination = export_paper(resolve_run(args.output_dir, args.run), args.export_dir)
        print(destination)
        return 0
    if args.command == "verify-environment":
        report = verify_environment()
        print(json.dumps(report, indent=2))
        return 0 if report.get("mamba_forward_backward") else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
