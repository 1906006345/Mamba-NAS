import time
import numpy as np
import torch
from .data import collate
from .training import isolated_rng


def profile_model(model, dataset, cfg):
    if not cfg["profiling"]["enabled"]:
        return {"status": "disabled"}
    c = cfg["profiling"]
    device = cfg["device"]
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else lambda: None
    # Deterministic representative: longest TRAIN sequence, batch=1. TEST is not
    # inspected to decide the profiling shape.
    example = max(range(len(dataset)), key=lambda i: len(dataset.samples[i]))
    def inputs(batch):
        x, mask, _, _ = collate([dataset[example]] * batch)
        return x.to(device), mask.to(device)
    with isolated_rng(), torch.no_grad():
        model.eval()
        x, mask = inputs(1)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for _ in range(c["warmup"]):
            model(x, mask)
        sync()
        times = []
        for _ in range(c["repeats"]):
            sync(); start = time.perf_counter(); model(x, mask); sync()
            times.append((time.perf_counter() - start) * 1000)
        x, mask = inputs(c["throughput_batch"])
        for _ in range(c["warmup"]):
            model(x, mask)
        sync(); start = time.perf_counter()
        for _ in range(c["repeats"]):
            model(x, mask)
        sync(); elapsed = time.perf_counter() - start
        return {"status": "complete", "backend": cfg["backend"], "batch1_median_ms": float(np.median(times)), "batch1_p90_ms": float(np.percentile(times, 90)), "throughput_samples_per_second": c["repeats"] * c["throughput_batch"] / elapsed, "throughput_batch": c["throughput_batch"], "input_length": len(dataset.samples[example]), "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None, "profiler_flops_status": "not_measured; analytic MAC estimate reported separately"}
