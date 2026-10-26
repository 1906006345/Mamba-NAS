"""Thin, Conda-compatible launchers. Run with the activated environment's Python.

python mamba_tree_nas/scripts/run.py primary --dry-run --with-refit
python mamba_tree_nas/scripts/run.py tensorboard --profile paper
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mamba_tree_nas.config import load_config, resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("verify", "smoke", "primary", "uea10", "comparisons", "download", "refit", "status", "report", "tensorboard", "test"))
    args, extra = p.parse_known_args()
    if args.action == "tensorboard":
        t = argparse.ArgumentParser()
        t.add_argument("--config")
        t.add_argument("--profile", default="paper")
        t.add_argument("--set", action="append", default=[])
        known, other = t.parse_known_args(extra)
        c = load_config(known.config, known.profile, known.set)
        command = [sys.executable, "-m", "tensorboard.main", "--logdir", str(resolve_path(c["output"]))] + other
    elif args.action == "test":
        command = [sys.executable, "-m", "pytest", "mamba_tree_nas/tests", "-q"] + extra
    else:
        mapping = {"verify": ["verify-environment"], "primary": ["run-suite", "--suite", "primary"], "uea10": ["run-suite", "--suite", "uea10"], "comparisons": ["run-suite", "--suite", "primary"]}
        command = [sys.executable, "-m", "mamba_tree_nas"] + mapping.get(args.action, [args.action]) + extra
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
