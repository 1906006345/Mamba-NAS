"""Measure independent classifiers under the frozen request workload."""
import math
import random
import statistics
import time

import numpy as np
import torch

from .architecture import Architecture
from .artifacts import write_csv
from .cost import macs
from .data import collate
from .deployment_metrics import cost_order_summary
from .deployment_protocol import bind
from .followup_evaluation import atomic_json
from .training import isolated_rng, make_model, seed_all


def _timed_ms(forward, synchronize, count):
    samples = []
    for _ in range(count):
        synchronize()
        start = time.perf_counter()
        forward()
        synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return samples


def _event_ms(forward, sync, count):
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(count):
        first.record()
        forward()
        last.record()
        sync()
        samples.append(first.elapsed_time(last))
    return samples


@torch.inference_mode()
def measure_round(study, arch):
    cfg, data = study.cfg, study.data
    settings = study.settings["benchmark"]
    device = torch.device(cfg["device"])
    is_cuda = device.type == "cuda"
    samples = []
    # This context also places CUDA events on the requested device.
    from contextlib import nullcontext

    with isolated_rng(), (torch.cuda.device(device) if is_cuda else nullcontext()):
        seed_all(study.settings["sampling_seed"], cfg["train"]["deterministic"], False)
        sync = (lambda: torch.cuda.synchronize(device)) if is_cuda else (lambda: None)
        sync()
        baseline = torch.cuda.memory_allocated(device) if is_cuda else 0
        model = make_model(arch, data, cfg).eval()
        parameters = sum(p.numel() for p in model.parameters())
        resident_model_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        for item in study.inputs:
            x_cpu, mask_cpu, _, _ = collate([data.train[item["position"]]])
            x, mask = x_cpu.to(device), mask_cpu.to(device)

            def forward():
                return model(x, mask)

            def request():
                return model(x_cpu.to(device), mask_cpu.to(device)).cpu()

            for _ in range(settings["warmup"]):
                forward()
            sync()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            wall = _timed_ms(forward, sync, settings["repeats"])
            peak = (torch.cuda.max_memory_allocated(device) - baseline if is_cuda else
                    resident_model_bytes + x.numel() * x.element_size() + mask.numel() * mask.element_size())
            events = _event_ms(forward, sync, settings["repeats"]) if is_cuda else []
            del x, mask
            for _ in range(settings["warmup"]):
                request()
            transfers = _timed_ms(request, sync, settings["repeats"])
            batch = settings["throughput_batch"]
            x_batch = x_cpu.expand(batch, -1, -1).contiguous().to(device)
            mask_batch = mask_cpu.expand(batch, -1).contiguous().to(device)
            for _ in range(settings["warmup"]):
                model(x_batch, mask_batch)
            sync()
            start = time.perf_counter()
            for _ in range(settings["repeats"]):
                model(x_batch, mask_batch)
            sync()
            throughput = batch * settings["repeats"] / (time.perf_counter() - start)
            del x_batch, mask_batch
            samples.append({**item, "wall_ms": wall, "cuda_event_ms": events, "request_ms": transfers,
                            "peak_bytes": peak, "throughput_samples_per_second": throughput,
                            "macs": macs(arch, item["length"], data.channels, len(data.classes))})
    for item in samples:
        values = item["wall_ms"] + item["request_ms"] + [item["peak_bytes"], item["throughput_samples_per_second"]]
        if not all(math.isfinite(v) and v > 0 for v in values) or not all(
                math.isfinite(v) and v >= 0 for v in item["cuda_event_ms"]):
            raise FloatingPointError("Invalid latency, peak allocation or throughput")
    return {"inputs": samples, "parameters": parameters, "memory_kind": (
        "cuda_peak_allocated_minus_pre_model_baseline" if is_cuda else "synthetic_model_plus_input_bytes"),
        "throughput_batch": settings["throughput_batch"]}


def get_round(study, arch, index):
    return study.cached("profile", {"hash": arch.hash, "round": index}, lambda: measure_round(study, arch))


def _summarize_rounds(rounds):
    inputs = []
    for index, template in enumerate(rounds[0]["inputs"]):
        rows = [r["inputs"][index] for r in rounds]

        def quantile(key, percentile):
            values = [v for row in rows for v in row[key]]
            return float(np.percentile(values, percentile)) if values else None

        inputs.append({k: template[k] for k in ("source_index", "length", "quantiles", "macs")} | {
            "latency_ms": quantile("wall_ms", 50), "latency_p95_ms": quantile("wall_ms", 95),
            "cuda_event_p50_ms": quantile("cuda_event_ms", 50), "cuda_event_p95_ms": quantile("cuda_event_ms", 95),
            "request_p50_ms": quantile("request_ms", 50), "request_p95_ms": quantile("request_ms", 95),
            "peak_bytes": max(r["peak_bytes"] for r in rows),
            "throughput_samples_per_second": statistics.mean(r["throughput_samples_per_second"] for r in rows),
        })
    medians = [statistics.mean(statistics.median(x["wall_ms"]) for x in r["inputs"]) for r in rounds]
    result = {k: statistics.mean(row[k] for row in inputs) if inputs[0][k] is not None else None for k in (
        "latency_ms", "latency_p95_ms", "cuda_event_p50_ms", "cuda_event_p95_ms", "request_p50_ms", "request_p95_ms",
        "throughput_samples_per_second")}
    return dict(result, peak_bytes=max(x["peak_bytes"] for x in inputs), inputs=inputs, round_latency_ms=medians,
                round_latency_sd_ms=statistics.stdev(medians) if len(medians) > 1 else 0.,
                parameters=rounds[0]["parameters"], memory_kind=rounds[0]["memory_kind"],
                throughput_batch=rounds[0]["throughput_batch"],
                elapsed_seconds=sum(r["elapsed_seconds"] for r in rounds))


def profile(study, arch):
    rounds, hits = [], 0
    for i in range(study.settings["benchmark"]["rounds"]):
        result, hit = get_round(study, arch, i)
        rounds.append(result)
        hits += hit
    return _summarize_rounds(rounds), hits


def benchmark(study):
    arches = {a.hash: a for a in study.panel}
    for pair in study.rate_pairs:
        arch = Architecture.from_dict(pair["edited"])
        arches[arch.hash] = arch
    errors = {}
    for index in range(study.settings["benchmark"]["rounds"]):
        order = sorted(arches)
        random.Random(study.settings["sampling_seed"] + index).shuffle(order)
        for key in order:
            if key in errors:
                continue
            try:
                get_round(study, arches[key], index)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                errors[key] = str(exc)
        print(f"[BENCHMARK] {study.data.name} round={index + 1} architectures={len(order)} errors={len(errors)}", flush=True)
    rows = {}
    panel_hashes = {a.hash for a in study.panel}
    for key, arch in arches.items():
        row = {"hash": key, "architecture": arch.to_dict(), "panel": key in panel_hashes, "macs": macs(arch, *study.dimensions)}
        if key in errors:
            rows[key] = dict(row, status="failed", error=errors[key])
        else:
            measurement, _ = profile(study, arch)
            rows[key] = dict(row, status="complete", **measurement)
    pairs = []
    for pair in study.rate_pairs:
        original = rows[Architecture.from_dict(pair["base"]).hash]
        edited = rows[Architecture.from_dict(pair["edited"]).hash]
        ok = all(x["status"] == "complete" for x in (original, edited))
        row = {k: v for k, v in pair.items() if k not in ("base", "edited")}
        row.update(base_hash=original["hash"], edited_hash=edited["hash"], status="complete" if ok else "failed")
        if ok:
            for name in ("macs", "latency_ms", "peak_bytes"):
                row["base_" + name], row["edited_" + name] = original[name], edited[name]
                row[name + "_delta"] = edited[name] - original[name]
        pairs.append(row)
    output = study.root / "benchmark"
    panel_rows = [rows[a.hash] for a in study.panel]
    write_csv(output / "architectures.csv", list(rows.values()))
    write_csv(output / "inputs.csv", [dict(item, hash=row["hash"]) for row in rows.values() for item in row.get("inputs", [])])
    write_csv(output / "rate_pairs.csv", pairs)
    summary = cost_order_summary(panel_rows)
    summary.update(status="complete" if summary["valid"] == len(study.panel) else "failed", rate_pairs=len(pairs),
                   valid_rate_pairs=sum(pair["status"] == "complete" for pair in pairs), errors=errors)
    atomic_json(output / "summary.json", summary)
    if summary["status"] != "complete":
        raise RuntimeError(f"Cannot freeze hardware scales with failed panel measurements: {output}")
    multiplier = study.settings["scale_multiplier"]
    bind(output / "scales.json", {"latency_ms": multiplier * max(r["latency_ms"] for r in panel_rows),
                                 "peak_bytes": multiplier * max(r["peak_bytes"] for r in panel_rows)})
