"""Scalar-only, policy-independent evaluation cache and fixed-budget audit runner.

This module is opt-in: the original search/training entry points are unchanged.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .architecture import Architecture
from .artifacts import atomic_json as _atomic_json, environment, read_json, run_lock
from .config import digest
from .training import evaluate, loader, make_model, seed_all


def atomic_json(path, value):
    # Windows file indexers can briefly open a newly replaced scalar file.
    # Retry the atomic operation, never downgrade it to an in-place write.
    for attempt in range(8):
        try:
            return _atomic_json(path, value)
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.025 * (attempt + 1))


def source_identity():
    root = Path(__file__).parent
    names = ("architecture.py", "model.py", "data.py", "cost.py", "metrics.py",
             "training.py", "proxy.py", "profiling.py", "refit.py", "artifacts.py",
             "config.py", "followup_evaluation.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def tensor_hash(values):
    result = hashlib.sha256()
    for name, tensor in values:
        value = tensor.detach().cpu().contiguous()
        result.update(str((name, str(value.dtype), list(value.shape))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def fixed_train(arch, data, cfg, seed, epochs, callback=None):
    """Same optimizer/leaf model, fixed epochs, best AND final validation.

    Hashes summarize tensors without saving weights, activations or gradients.
    Sample-order hashes are accumulated from the actual training iterator.
    """
    if cfg["backend"] == "test" and not data.synthetic:
        raise ValueError("Test mixer is only allowed for synthetic smoke data")
    seed_all(seed, cfg["train"]["deterministic"], cfg["runtime"]["allow_tf32"])
    arch = Architecture.from_dict(arch.to_dict())
    model = make_model(arch, data, cfg)
    initial = tensor_hash(model.state_dict().items())
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                 weight_decay=train_cfg["weight_decay"])
    batches = loader(data.train, cfg, seed, True)
    history, best_key, best, best_epoch = [], None, None, None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order, count, total, correct, trace = hashlib.sha256(), 0, 0.0, 0, {}
        for index, (x, mask, y, ids) in enumerate(batches):
            order.update(ids.numpy().tobytes())
            x, mask, y = x.to(cfg["device"]), mask.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fixed-budget CE")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"], error_if_nonfinite=True)
            if index == 0:
                trace = {"first_logits_hash": tensor_hash([("logits", logits)]),
                         "first_loss": float(loss.detach()),
                         "first_gradient_hash": tensor_hash((n, p.grad) for n, p in model.named_parameters() if p.grad is not None),
                         "first_batch_ids_hash": tensor_hash([("ids", ids)])}
            optimizer.step()
            if index == 0:
                trace["first_update_hash"] = tensor_hash(model.state_dict().items())
            total += loss.item() * len(y)
            count += len(y)
            correct += int((logits.argmax(-1) == y).sum())
        validation, _ = evaluate(model, data.val, cfg, len(data.classes))
        key = (-validation["macro_f1"], validation["ce"], epoch)
        if best_key is None or key < best_key:
            best_key, best, best_epoch = key, validation, epoch
        row = dict(epoch=epoch, train_ce=total / count, train_accuracy=correct / count,
                   sample_order_hash=order.hexdigest(), **trace,
                   **{"validation_" + k: validation[k] for k in ("macro_f1", "ce", "accuracy")})
        history.append(row)
        if callback:
            callback(row)
    result = {"status": "complete", "initial_weights_hash": initial,
              "final_weights_hash": tensor_hash(model.state_dict().items()),
              "validation": best, "final_validation": validation,
              "best_epoch": best_epoch, "epochs_run": epochs, "history": history,
              "training_seconds": time.perf_counter() - start,
              "parameters": sum(p.numel() for p in model.parameters()), "test_accessed": False}
    del optimizer, model
    return result


class EvaluationCache:
    def __init__(self, root, cfg, data):
        self.root, self.cfg, self.data = Path(root), cfg, data
        self.protocol = {"version": 1, "data": data.manifest, "train": cfg["train"],
                         "backend": cfg["backend"], "device": cfg["device"], "runtime": cfg["runtime"],
                         "environment": environment(), "source": source_identity(),
                         "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}

    def identity(self, arch, seed, mode="candidate", epochs=None):
        if mode not in ("candidate", "diagnostic") or (mode == "diagnostic" and not epochs):
            raise ValueError("Invalid evaluation mode/epochs")
        identity = dict(protocol=self.protocol, architecture=arch.hash, seed=seed, mode=mode, epochs=epochs)
        if mode == "candidate":
            identity.update(proxy=self.cfg["proxy"], proxy_seeds=self.cfg["seeds"]["proxy"])
        return identity

    def measure(self, arch, seed, owner, mode="candidate", epochs=None, callback=None):
        if self.cfg["backend"] == "test" and not self.data.synthetic:
            raise ValueError("Test mixer is only allowed for synthetic smoke data")
        identity = self.identity(arch, seed, mode, epochs)
        key = digest(identity)
        path = self.root / self.data.name / f"{key}.json"
        start = time.perf_counter()
        with run_lock(path.with_suffix(".lock")):
            if path.exists():
                saved = read_json(path)
                if saved["identity"] != identity or saved["result"]["status"] != "complete":
                    raise ValueError(f"Evaluation cache identity/status mismatch: {path}")
                hit = True
            else:
                if mode == "candidate":
                    from .proxy import measure_proxy
                    from .training import train
                    proxy = measure_proxy(arch, self.data, self.cfg)
                    trained = train(arch, self.data, self.cfg, seed, epoch_callback=callback)
                    result = {k: trained[k] for k in ("validation", "best_epoch", "epochs_run", "training_seconds",
                                                     "training_peak_allocated_gpu_bytes", "parameters", "history")}
                    result.update(status="complete", proxy=proxy)
                    del trained
                else:
                    result = fixed_train(arch, self.data, self.cfg, seed, epochs, callback)
                saved = {"identity": identity, "first_owner": owner, "result": result}
                atomic_json(path, saved)
                hit = False
        return deepcopy(saved["result"]), {"key": key, "hit": hit, "first_owner": saved["first_owner"],
                                            "actual_seconds": time.perf_counter() - start,
                                            "path": str(path.resolve())}
