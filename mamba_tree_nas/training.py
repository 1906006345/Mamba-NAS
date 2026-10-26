from __future__ import annotations

from contextlib import contextmanager
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import collate
from .metrics import classification
from .model import Classifier


def seed_all(seed, deterministic=True, allow_tf32=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    # Official fused scan may not expose a deterministic implementation. Record
    # warn-only behavior in docs; seeds alone do not promise bitwise CUDA replay.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


@contextmanager
def isolated_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
        torch.use_deterministic_algorithms(state[4], warn_only=state[5])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = state[6:8]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = state[8:]


def make_model(arch, data, cfg):
    if cfg["backend"] == "test":
        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    return Classifier(arch, data.channels, len(data.classes), cfg["train"]["dropout"], cfg["backend"], cfg["device"]).to(cfg["device"])


def loader(dataset, cfg, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, num_workers=cfg["train"]["num_workers"], collate_fn=collate, generator=generator)


def evaluate(model, dataset, cfg, classes):
    model.eval()
    labels, probs, indices = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for x, mask, y, ids in loader(dataset, cfg, 0, False):
            x, mask, y = x.to(cfg["device"]), mask.to(cfg["device"]), y.to(cfg["device"])
            logits = model(x, mask)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits")
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            labels.extend(y.cpu().tolist()); probs.append(logits.softmax(-1).cpu().numpy()); indices.extend(ids.tolist())
    probabilities = np.concatenate(probs)
    metrics = classification(labels, probabilities, classes, loss_sum / len(labels))
    return metrics, {"sample_index": np.array(indices), "label": np.array(labels), "prediction": probabilities.argmax(1), "probabilities": probabilities.astype(np.float32)}


def train(arch, data, cfg, seed, train_data=None, validation=None, fixed_epochs=None, epoch_callback=None):
    seed_all(seed, cfg["train"]["deterministic"], cfg["runtime"]["allow_tf32"])
    model = make_model(arch, data, cfg)
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    train_data = train_data if train_data is not None else data.train
    if fixed_epochs is None:
        validation = data.val if validation is None else validation
    config = cfg["train"]
    epochs = fixed_epochs if fixed_epochs is not None else config["epochs"]
    dataloader = loader(train_data, cfg, seed, True)
    best_key, best_state, best_metrics, best_epoch = None, None, None, 0
    history, stale = [], 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, correct, count = 0.0, 0, 0
        for x, mask, y, _ in dataloader:
            x, mask, y = x.to(cfg["device"]), mask.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite supervised CE")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_total += loss.item() * len(y); correct += int((logits.argmax(-1) == y).sum()); count += len(y)
        row = {"epoch": epoch, "train_ce": loss_total / count, "train_accuracy": correct / count}
        if fixed_epochs is None:
            metrics, _ = evaluate(model, validation, cfg, len(data.classes))
            row.update({"validation_" + key: metrics[key] for key in ("ce", "macro_f1", "accuracy", "balanced_accuracy")})
            key = (-metrics["macro_f1"], metrics["ce"], epoch)
            if best_key is None or key < best_key:
                best_key, best_metrics, best_epoch = key, metrics, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(row)
        if epoch_callback:
            epoch_callback(row)
        if fixed_epochs is None and epoch >= config["min_epochs"] and stale >= config["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, best_state
    return {"model": model, "history": history, "validation": best_metrics, "best_epoch": best_epoch if fixed_epochs is None else epochs, "epochs_run": len(history), "training_seconds": time.perf_counter() - start, "parameters": sum(p.numel() for p in model.parameters()), "training_peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if str(cfg["device"]).startswith("cuda") else None}
