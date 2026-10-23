from __future__ import annotations

import numpy as np


def profile_cuda(model, values, mask, warmups: int = 20, iterations: int = 100) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmups):
            model(values, mask)
        torch.cuda.synchronize()
        timings = []
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(values, mask)
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))
    profile_flops = None
    profile_status = "unsupported_or_incomplete"
    try:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
        )
        profiler.start()
        with torch.no_grad():
            model(values, mask)
        profiler.stop()
        total_flops = sum(float(getattr(event, "flops", 0) or 0) for event in profiler.key_averages())
        if total_flops > 0:
            profile_flops = int(total_flops)
            profile_status = "available"
    except Exception:
        profile_status = "unsupported_or_incomplete"
    median = float(np.median(timings))
    return {
        "device": torch.cuda.get_device_name(values.device),
        "batch_size": int(values.shape[0]),
        "sequence_length": int(values.shape[1]),
        "input_channels": int(values.shape[2]),
        "warmups": warmups,
        "iterations": iterations,
        "latency_median_ms": median,
        "latency_p90_ms": float(np.percentile(timings, 90)),
        "throughput_samples_per_sec": float(1000.0 * values.shape[0] / median) if median > 0 else None,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "profile_flops": profile_flops,
        "profile_macs": int(profile_flops // 2) if profile_flops is not None else None,
        "profile_flops_status": profile_status,
        "latency_definition": "CUDA event median/p90; warmup=20; batch=1",
        "memory_definition": "torch.cuda.max_memory_allocated during profiling",
    }
