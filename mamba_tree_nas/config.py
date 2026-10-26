from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def merge(base, updates):
    out = copy.deepcopy(base)
    for key, value in updates.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def load_config(path=None, profile="paper", overrides=()):
    import yaml
    source = Path(path or DEFAULT_CONFIG).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if profile not in raw["profiles"]:
        raise ValueError(f"Unknown profile: {profile}")
    cfg = merge(raw["defaults"], raw["profiles"][profile])
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError("--set requires dotted.path=YAML_value")
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown configuration path: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    cfg["profile"] = profile
    cfg["config_version"] = raw["version"]
    validate(cfg)
    return cfg


def validate(c):
    if c["backend"] not in ("test", "mamba"):
        raise ValueError("backend must be mamba or test")
    if c["train"]["precision"] != "fp32":
        raise ValueError("This protocol implements FP32 only")
    if not 0 < c["data"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if not 1 <= c["search"]["population"] <= c["search"]["budget"]:
        raise ValueError("Require 1 <= population <= budget")
    if c["space"]["max_leaves"] < 1 or c["space"]["max_depth"] < 0:
        raise ValueError("Invalid tree bounds")
    # RES(OP) chains require n internal levels for n >= 2.
    required = 1 if c["space"]["max_leaves"] == 1 else c["space"]["max_leaves"]
    if c["space"]["max_depth"] < required:
        raise ValueError("Depth bound cannot represent the longest residual chain")
    for budget in (c["train"], merge(c["train"], c["calibration"])):
        if not 1 <= budget["min_epochs"] <= budget["epochs"] or budget["patience"] < 1:
            raise ValueError("Invalid epoch/patience budget")
    for name in ("sampling",):
        if any(v < 0 for v in c["space"][name].values()) or abs(sum(c["space"][name].values()) - 1) > 1e-8:
            raise ValueError("Sampling probabilities must sum to 1")
    if any(x <= 1 for x in c["search"]["hv_reference"]):
        raise ValueError("HV reference must dominate the fixed unit bounds")
    if c["runtime"]["cpu_threads"] < 1 or c["space"]["max_attempts"] < 1:
        raise ValueError("Thread and sampling-attempt counts must be positive")
    for key in ("crossover_probability", "mutation_probability"):
        if not 0 <= c["search"][key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if any(v < 0 for v in c["search"]["mutation_weights"].values()) or abs(sum(c["search"]["mutation_weights"].values()) - 1) > 1e-8:
        raise ValueError("Mutation weights must sum to 1")
    if c["proxy"]["epsilon"] <= 0 or c["data"]["normalization_epsilon"] <= 0:
        raise ValueError("Numerical epsilons must be positive")
    for key in ("search", "proxy", "refit"):
        values = c["seeds"][key]
        if not values or len(values) != len(set(values)) or any(not isinstance(v, int) for v in values):
            raise ValueError(f"{key} seeds must be a nonempty unique integer list")
    if c["profiling"]["repeats"] < 1 or c["profiling"]["throughput_batch"] < 1:
        raise ValueError("Profiling repeats/batch must be positive")
    if c["train"]["batch_size"] < 1 or not 0 <= c["train"]["dropout"] < 1 or c["train"]["learning_rate"] <= 0:
        raise ValueError("Invalid optimizer/batch/dropout settings")
    if not 0 < c["report"]["top_fraction"] <= 1:
        raise ValueError("report.top_fraction must be in (0,1]")


def resolve_path(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def protocol_fingerprint(cfg):
    # Excludes search randomness, logging and output location. Cache remains run-local.
    keys = ("config_version", "backend", "runtime", "data", "space", "train", "calibration", "proxy", "seeds")
    return digest({k: cfg[k] for k in keys})
