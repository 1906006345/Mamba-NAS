"""Unified model size, FLOPs and runtime measurements."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np


def trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _forward(model, values, padding_mask):
    """Support both Mamba-NAS ``(x, mask)`` and legacy experiment signatures."""
    try:
        return model(values, padding_mask, None, None)
    except TypeError:
        return model(values, padding_mask)


def _first_batch(loader: Iterable):
    batch = next(iter(loader))
    if len(batch) == 3:
        values, _, padding_mask = batch
    else:
        values, padding_mask = batch[0], batch[-1]
    return values[:1], padding_mask[:1]


def profile_forward(model, loader: Iterable, device, warmups: int = 20, iterations: int = 100) -> dict:
    import torch

    values, padding_mask = _first_batch(loader)
    values = values.float().to(device)
    padding_mask = padding_mask.to(device)
    was_training = model.training
    model.eval()
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmups):
            _forward(model, values, padding_mask)
        if use_cuda:
            torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _forward(model, values, padding_mask)
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))
        else:
            started = time.perf_counter()
            with torch.no_grad():
                _forward(model, values, padding_mask)
            timings.append((time.perf_counter() - started) * 1000.0)

    profile_flops = None
    profile_status = "unsupported_or_incomplete"
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities, with_flops=True)
        profiler.start()
        with torch.no_grad():
            _forward(model, values, padding_mask)
        profiler.stop()
        total_flops = sum(float(getattr(event, "flops", 0) or 0) for event in profiler.key_averages())
        if total_flops > 0:
            profile_flops = int(total_flops)
            profile_status = "available"
    except Exception:
        profile_status = "unsupported_or_incomplete"

    if was_training:
        model.train()
    latency = float(np.median(timings)) if timings else float("nan")
    p90 = float(np.percentile(timings, 90)) if timings else float("nan")
    memory = int(torch.cuda.max_memory_allocated()) if use_cuda else None
    return {
        "profile_flops": profile_flops,
        "profile_macs": int(profile_flops // 2) if profile_flops is not None else None,
        "profile_flops_status": profile_status,
        "latency_median_ms": latency,
        "latency_p90_ms": p90,
        "throughput_samples_per_sec": float(1000.0 / latency) if latency > 0 else None,
        "peak_memory_bytes": memory,
        "device": str(device),
        "batch_size": 1,
        "sequence_length": int(values.shape[1]),
        "input_channels": int(values.shape[2]),
        "warmups": int(warmups),
        "iterations": int(iterations),
        "latency_definition": "CUDA event median/p90; CPU perf_counter fallback",
        "memory_definition": "torch.cuda.max_memory_allocated during profiling",
    }


def forward_macs(model, loader: Iterable, device) -> int:
    """Backward-compatible profiler MAC accessor."""
    value = profile_forward(model, loader, device, warmups=0, iterations=1)["profile_macs"]
    if value is None:
        raise RuntimeError("Profiler did not expose forward FLOPs for this model")
    return int(value)


def model_performance(model, loader: Iterable, device) -> dict:
    runtime = profile_forward(model, loader, device)
    return {
        "parameters": trainable_parameters(model),
        "macs": runtime["profile_macs"],
        "flops": runtime["profile_flops"],
        "profile_macs": runtime["profile_macs"],
        "profile_flops": runtime["profile_flops"],
        **runtime,
        "macs_definition": "torch.profiler forward FLOPs / 2, batch=1, first evaluation shape",
        "flops_definition": "torch.profiler forward FLOPs, batch=1, first evaluation shape",
        "efficiency_definition": "profiled forward compute plus batch=1 runtime",
    }
