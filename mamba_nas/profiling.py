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
    return {
        "device": torch.cuda.get_device_name(values.device),
        "batch_size": int(values.shape[0]),
        "sequence_length": int(values.shape[1]),
        "input_channels": int(values.shape[2]),
        "warmups": warmups,
        "iterations": iterations,
        "latency_median_ms": float(np.median(timings)),
        "latency_p90_ms": float(np.percentile(timings, 90)),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }

