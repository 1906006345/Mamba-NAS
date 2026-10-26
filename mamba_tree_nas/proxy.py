from __future__ import annotations

import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import collate
from .training import isolated_rng, make_model, seed_all
from .artifacts import error_record


def gradient_statistics(named_parameters, epsilon):
    # Accumulate in float64, release tensors after each seed. No tensors exported.
    theta2 = grad2 = abs_product = dot = 0.0
    count = 0
    missing, nonfinite, near_zero = [], [], []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name); continue
        p, g = parameter.detach().double(), parameter.grad.detach().double()
        if not torch.isfinite(p).all() or not torch.isfinite(g).all():
            nonfinite.append(name); continue
        theta2 += p.square().sum().item(); grad2 += g.square().sum().item()
        abs_product += (p * g).abs().sum().item(); dot += (p * g).sum().item(); count += p.numel()
        if g.norm().item() <= epsilon:
            near_zero.append(name)
    if nonfinite or missing or count == 0:
        status = "nonfinite" if nonfinite else "missing_gradient"
    elif grad2 <= epsilon ** 2 or theta2 <= epsilon ** 2:
        status = "near_zero"
    else:
        status = "ok"
    l2, theta_norm = math.sqrt(grad2), math.sqrt(theta2)
    alignment = abs(dot) / max(theta_norm * l2, epsilon)
    # Defined explicitly for reproducibility: gradient energy times absolute
    # parameter-gradient cosine, in [gradient units]. Not a validated ranking.
    product = l2 * alignment
    return {"status": status, "parameter_count": count, "gradient_l2": l2 if not nonfinite else None, "gradient_rms": l2 / math.sqrt(count) if count and not nonfinite else None, "sum_abs_gtheta": abs_product if not nonfinite else None, "mean_abs_gtheta": abs_product / count if count and not nonfinite else None, "absolute_cosine": alignment if not nonfinite else None, "product": product if status == "ok" else None, "missing_gradients": missing, "nonfinite_parameters": nonfinite, "near_zero_parameters": near_zero}


def measure_proxy(arch, data, cfg):
    if not cfg["proxy"]["enabled"]:
        return {"status": "disabled", "seconds": 0.0}
    start = time.perf_counter()
    repeats = []
    try:
        with isolated_rng():
            x, mask, y, ids = collate([data.train[i] for i in data.proxy_indices])
            x, mask, y = x.to(cfg["device"]), mask.to(cfg["device"]), y.to(cfg["device"])
            for seed in cfg["seeds"]["proxy"]:
                seed_all(seed, cfg["train"]["deterministic"], cfg["runtime"]["allow_tf32"])
                model = make_model(arch, data, cfg)
                model.eval()
                loss = F.cross_entropy(model(x, mask), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite proxy CE")
                loss.backward()
                repeats.append({"seed": seed, "ce": loss.item(), "full": gradient_statistics(model.named_parameters(), cfg["proxy"]["epsilon"]), "trunk": gradient_statistics(model.tree.named_parameters(), cfg["proxy"]["epsilon"])})
                del model, loss
            aggregate = {}
            for scope in ("full", "trunk"):
                aggregate[scope] = {}
                for field in ("gradient_l2", "gradient_rms", "sum_abs_gtheta", "mean_abs_gtheta", "absolute_cosine", "product"):
                    values = [r[scope][field] for r in repeats]
                    aggregate[scope][field] = float(np.mean(values)) if values and all(v is not None and math.isfinite(v) for v in values) else None
                product = aggregate[scope]["product"]
                aggregate[scope]["log10_mean_product"] = math.log10(product) if product is not None and product > cfg["proxy"]["epsilon"] else None
            return {"status": "complete", "repeats": repeats, "aggregate": aggregate, "mean_ce": float(np.mean([r["ce"] for r in repeats])), "seconds": time.perf_counter() - start, "sample_indices": ids.tolist(), "formula": "product=gradient_l2*abs(dot(theta,g))/(||theta||2*||g||2+epsilon clamp); log10_mean_product=log10(mean_seed(product))", "flags": [f"{r['seed']}:{s}:{r[s]['status']}" for r in repeats for s in ("full", "trunk") if r[s]["status"] != "ok"]}
    except Exception as exc:
        return {"status": "failed", "error": error_record(exc, "proxy"), "repeats": repeats, "seconds": time.perf_counter() - start}
