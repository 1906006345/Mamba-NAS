"""Shared input panel, protocol manifests and resumable scalar caches."""
import hashlib
import math
import os
import random
import socket
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .architecture import Architecture, Space, at, paths, substitute
from .artifacts import environment, error_record, implementation_fingerprint, read_json, run_lock
from .config import DEFAULT_CONFIG, digest, load_config, merge, validate
from .cost import macs
from .data import DataProtocol
from .followup_evaluation import atomic_json


def configuration(profile, path=None, data_root=None):
    import yaml

    cfg = load_config(path, profile, multiscale=True)
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))["deployment_nas"]
    settings = merge({key: value for key, value in raw.items() if key != "profiles"}, raw["profiles"][profile])
    cfg["deployment_nas"] = settings
    cfg["search"].update(settings["search"])
    if data_root is not None:
        cfg["data"]["root"] = str(Path(data_root).resolve())
    if profile == "smoke":
        cfg["logging"]["tensorboard"] = False
        cfg["smoke"]["length"] = 65
    validate(cfg)
    if (not settings["checkpoints"] or settings["checkpoints"] != sorted(set(settings["checkpoints"]))
            or settings["checkpoints"][0] < 1
            or settings["checkpoints"][-1] != cfg["search"]["budget"]):
        raise ValueError("Checkpoints must increase and end at the search budget")
    for name in ("search_seeds", "training_seeds"):
        seeds = settings[name]
        if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError(f"Invalid {name}")
    for key in ("panel_size", "reference_epochs", "supernet_epochs", "audit_quota"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid deployment {key}")
    if settings["rate_pairs"] < 0 or settings["scale_multiplier"] <= 1:
        raise ValueError("Invalid rate-pair count or hardware scale multiplier")
    if settings["reference_epochs"] < cfg["train"]["epochs"]:
        raise ValueError("Reference training must cover at least the short training budget")
    if len(settings["hv_reference"]) != 3 or not all(math.isfinite(x) and x > 1 for x in settings["hv_reference"]):
        raise ValueError("Deployment HV requires three reference coordinates above one")
    if any(type(v) is not int or v < 1 for v in settings["benchmark"].values()):
        raise ValueError("Benchmark iteration and batch counts must be positive")
    if not settings["datasets"] or set(settings["datasets"].values()) - set(cfg["experiments"]):
        raise ValueError("Invalid deployment dataset and experiment mapping")
    return cfg


def runtime_identity(cfg):
    import torch
    from .model import mixer_factory

    info = environment()
    if cfg["profile"] == "smoke":
        if cfg["backend"] != "test" or cfg["device"] != "cpu":
            raise ValueError("Deployment smoke requires the synthetic CPU mixer")
        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        return dict(info, mode="synthetic_cpu_smoke")
    if cfg["backend"] != "mamba" or not torch.cuda.is_available() or not cfg["device"].startswith("cuda"):
        raise RuntimeError("Deployment experiments require official Mamba and an NVIDIA L20")
    device = torch.device(cfg["device"])
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    target = cfg["deployment_nas"]["target_gpu"].casefold()
    if target not in properties.name.casefold().split():
        raise RuntimeError(f"Expected target {target.upper()}, found {properties.name}")
    mixer_factory(cfg["backend"], cfg["device"])
    # Include driver and physical inventory even on PyTorch builds without UUID properties.
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.splitlines()
    if not inventory:
        raise RuntimeError("nvidia-smi returned no GPU identities")
    return dict(info, mode="cuda", gpu=properties.name, cuda_index=index,
                cuda_device_uuid=str(getattr(properties, "uuid", "")), driver_inventory=inventory,
                host=socket.gethostname(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def bind(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Cannot resume with a different deployment protocol: {path}")
    else:
        atomic_json(path, value)


def bind_study(output, cfg, runtime):
    manifest = output / "protocol.json"
    if not manifest.exists() and any(p.name != ".deployment.lock" for p in output.iterdir()):
        raise ValueError("Use an empty output folder or an existing deployment study")
    directory = Path(__file__).parent
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("deployment_*.py"))}
    identity = {"config": cfg, "runtime": runtime, "core": implementation_fingerprint(), "code": code}
    bind(manifest, identity)
    return digest(identity)


def select_inputs(dataset):
    lengths = [len(x) for x in dataset.samples]
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], dataset.indices[i]))
    selected = {}
    for label, quantile in (("p10", .1), ("p50", .5), ("p90", .9), ("max", 1.)):
        length = lengths[order[int(quantile * (len(order) - 1) + .5)]]
        pos = next(i for i in order if lengths[i] == length)
        selected.setdefault(pos, {"position": pos, "source_index": dataset.indices[pos],
                                  "length": length, "quantiles": []})["quantiles"].append(label)
    return list(selected.values())


def sample_panel(space, dimensions, count, seed):
    rng = random.Random(seed)
    strata = defaultdict(dict)
    for _ in range(max(512, count * 32)):
        arch = space.sample(rng)
        strata[arch.tree.leaves][arch.hash] = arch
    if sum(len(s) for s in strata.values()) < count:
        raise ValueError("Could not sample enough distinct panel architectures")
    allocations = dict.fromkeys(strata, 0)
    for _ in range(count):
        available = [k for k, a in allocations.items() if a < len(strata[k])]
        allocations[min(available, key=lambda k: (allocations[k], k))] += 1
    panel = []
    for leaves, quota in sorted(allocations.items()):
        candidates = sorted(strata[leaves].values(), key=lambda a: (macs(a, *dimensions), a.hash))
        if quota == 1:
            panel.append(candidates[len(candidates) // 2])
        elif quota:
            panel.extend(candidates[round(i * (len(candidates) - 1) / (quota - 1))] for i in range(quota))
    return panel


def sample_rate_pairs(space, panel, count):
    if not count:
        return []
    result, seen = [], set()
    for base in panel:
        for path in paths(base.tree):
            node = at(base.tree, path)
            if node.op != "OP":
                continue
            rates = space.rates(base.tokenizer)
            i = rates.index(node.leaf.rate)
            for index in (i - 1, i + 1):
                if not 0 <= index < len(rates):
                    continue
                edited = replace(node, leaf=replace(node.leaf, rate=rates[index]))
                child = replace(base, tree=substitute(base.tree, path, edited))
                pair = tuple(sorted((base.hash, child.hash)))
                if pair in seen:
                    continue
                seen.add(pair)
                result.append({"base": base.to_dict(), "edited": child.to_dict(), "leaf_path": list(path),
                               "base_rate": node.leaf.rate, "edited_rate": edited.leaf.rate})
                if len(result) == count:
                    return result
    return result


class Study:
    def __init__(self, cfg, output, dataset, fingerprint):
        self.cfg, self.settings = cfg, cfg["deployment_nas"]
        self.root = output / dataset
        self.data = DataProtocol(cfg, dataset, synthetic=cfg["profile"] == "smoke")
        self.dimensions = (self.data.max_length, self.data.channels, len(self.data.classes))
        experiment = cfg["experiments"][self.settings["datasets"][dataset]]
        self.space = Space(cfg["space"], experiment, min(map(len, self.data.train.samples)))
        self.identity = digest({"study": fingerprint, "data": self.data.fingerprint})
        bind(self.root / "data.json", self.data.manifest)
        inputs = select_inputs(self.data.train)
        panel_path = self.root / "panel.json"
        if panel_path.exists():
            panel = read_json(panel_path)
            if panel["identity"] != self.identity or panel["inputs"] != inputs:
                raise ValueError("Deployment architecture or input panel changed")
        else:
            arches = sample_panel(self.space, self.dimensions, self.settings["panel_size"], self.settings["sampling_seed"])
            # Serialize first so leaf paths in rate-pair metadata use the canonical tree.
            arches = [Architecture.from_dict(a.to_dict()) for a in arches]
            panel = {"identity": self.identity, "inputs": inputs, "architectures": [a.to_dict() for a in arches],
                     "rate_pairs": sample_rate_pairs(self.space, arches, self.settings["rate_pairs"])}
            bind(panel_path, panel)
        self.inputs = inputs
        self.panel = [Architecture.from_dict(a) for a in panel["architectures"]]
        self.rate_pairs = panel["rate_pairs"]

    def cached(self, kind, key, compute):
        identity = {"protocol": self.identity, "kind": kind, **key}
        path = self.root / "cache" / kind / (digest(identity) + ".json")
        if path.exists():
            result = read_json(path)
            if result["identity"] != identity:
                raise ValueError(f"Cache identity mismatch: {path}")
            return result["result"], True
        begin = time.perf_counter()
        try:
            result = dict(compute(), elapsed_seconds=time.perf_counter() - begin)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            atomic_json(path.with_suffix(".error.json"), error_record(exc, kind))
            raise
        atomic_json(path, {"identity": identity, "result": result})
        return result, False

    def standalone(self, arch, seed, mode):
        from .training import train

        if mode not in ("short", "reference"):
            raise ValueError(f"Unknown training budget: {mode}")

        def fit():
            cfg = self.cfg
            if mode == "reference":
                epochs = self.settings["reference_epochs"]
                # Validate every epoch, but guarantee the entire reference training budget.
                cfg = merge(cfg, {"train": {"epochs": epochs, "min_epochs": epochs, "patience": epochs}})
            result = train(arch, self.data, cfg, seed)
            return {k: result[k] for k in ("validation", "epochs_run", "best_epoch", "training_seconds")}

        return self.cached(mode, {"hash": arch.hash, "seed": seed}, fit)
