"""Single-path sharing for the same bounded tree grammar as standalone search."""
import os
import random
import tempfile
import time
from collections import Counter
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .architecture import Architecture
from .artifacts import read_json
from .data import collate
from .followup_evaluation import atomic_json
from .model import Classifier, RMSNorm, Tokenizer, TreeModule, mixer_factory
from .training import loader, seed_all


class SharedTreeView(TreeModule):
    """Ephemeral module view with the exact standalone forward and state keys."""
    def __init__(self, bank, node, width, slots, active):
        nn.Module.__init__(self)
        self.op = node.op
        self.branches = nn.ModuleList(SharedTreeView(bank, c, width, slots, active) for c in node.children)
        if node.op == "OP":
            slot = next(slots)
            leaf = node.leaf
            self.direction, self.rate = leaf.direction, leaf.rate
            norm = bank.key(slot, width)
            mixer = bank.key(slot, width, leaf.d_state, leaf.d_conv, leaf.expand)
            self.norm, self.mixer = bank.norms[norm], bank.mixers[mixer]
            self.dropout = nn.Dropout(bank.dropout)
            active.extend((f"norms/{norm}", f"mixers/{mixer}"))


class SharedClassifierView(Classifier):
    def __init__(self, bank, arch):
        nn.Module.__init__(self)
        self.arch = arch
        tokenizer = bank.key(arch.d_model, arch.tokenizer)
        head = bank.key(arch.d_model, arch.pooling)
        self.tokenizer, self.head = bank.tokenizers[tokenizer], bank.heads[head]
        self.activated = [f"tokenizers/{tokenizer}", f"heads/{head}"]
        self.tree = SharedTreeView(bank, arch.tree, arch.d_model, iter(range(bank.max_leaves)), self.activated)
        self.train(bank.training)


class TreeSupernet(nn.Module):
    def __init__(self, study):
        super().__init__()
        cfg, space = study.cfg, study.space
        self.space = space
        self.channels, self.classes = study.data.channels, len(study.data.classes)
        self.backend, self.device_name = cfg["backend"], cfg["device"]
        self.factory = mixer_factory(self.backend, self.device_name)
        self.dropout = cfg["train"]["dropout"]
        self.max_leaves = space.c["max_leaves"]
        widths = space.c["d_model"]
        self.tokenizers = nn.ModuleDict({
            self.key(w, t): Tokenizer(self.channels, w, t) for w, t in product(widths, space.c["tokenizer"])})
        self.heads = nn.ModuleDict({
            self.key(w, p): nn.Linear(w * (2 if p == "meanmax" else 1), self.classes)
            for w, p in product(widths, space.c["pooling"])})
        self.norms = nn.ModuleDict({
            self.key(s, w): RMSNorm(w) for s, w in product(range(self.max_leaves), widths)})
        self.mixers = nn.ModuleDict({
            self.key(s, w, z, c, e): self.factory(d_model=w, d_state=z, d_conv=c, expand=e)
            for s, w, z, c, e in product(range(self.max_leaves), widths, space.c["d_state"],
                                          space.c["d_conv"], space.c["expand"])})

    @staticmethod
    def key(*parts):
        return "_".join(map(str, parts))

    def path(self, arch):
        if not self.space.valid(arch):
            raise ValueError("Invalid supernet path")
        return SharedClassifierView(self, Architecture.from_dict(arch.to_dict()))

    def export(self, arch):
        path = self.path(arch)
        model = Classifier(path.arch, self.channels, self.classes, self.dropout, self.backend,
                           self.device_name, factory=self.factory)
        model.load_state_dict(path.state_dict(), strict=True)
        return model.eval()

    def activation_counts(self, arch, counts):
        return {name: counts.get(name, 0) for name in self.path(arch).activated}


def _save_checkpoint(path, checkpoint):
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(checkpoint, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@torch.inference_mode()
def verify_export(bank, arch, study):
    device = next(bank.parameters()).device
    x, mask, _, _ = collate([study.data.train[study.inputs[0]["position"]]])
    x, mask = x.to(device), mask.to(device)
    bank.eval()
    expected = bank.path(arch)(x, mask)
    actual = bank.export(arch).to(device)(x, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return float((actual - expected).abs().max())


def train_supernet(study, seed):
    """Resume at committed epoch boundaries and return the final bank on CPU."""
    started = time.perf_counter()
    cfg, train = study.cfg, study.cfg["train"]
    output = study.root / "supernet" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "checkpoint.pt"
    summary_path = output / "summary.json"
    seed_all(seed, train["deterministic"], cfg["runtime"]["allow_tf32"])
    bank = TreeSupernet(study)
    target = study.settings["supernet_epochs"]
    state = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
    if state is not None:
        if state["protocol"] != study.identity or state["seed"] != seed:
            raise ValueError("Supernet checkpoint protocol mismatch")
        bank.load_state_dict(state["model"])
    if state is not None and state["epoch"] == target and summary_path.exists():
        return bank.eval(), read_json(summary_path)
    bank = bank.to(cfg["device"])
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
    counts, history = Counter(), []
    completed, previous_seconds = 0, 0.
    if state is not None:
        completed, previous_seconds = state["epoch"], state["elapsed_seconds"]
        counts.update(state["counts"])
        history = state["history"]
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    del state
    for epoch in range(completed + 1, target + 1):
        bank.train()
        losses = []
        for x, mask, y, _ in loader(study.data.train, cfg, seed + epoch, True):
            arch = study.space.sample(rng)
            active = bank.path(arch)
            counts.update(active.activated)
            optimizer.zero_grad(set_to_none=True)
            logits = active(x.to(cfg["device"]), mask.to(cfg["device"]))
            loss = F.cross_entropy(logits, y.to(cfg["device"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite supernet training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(active.parameters(), train["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "training_loss": sum(losses) / len(losses), "batches": len(losses)})
        _save_checkpoint(path, {
            "protocol": study.identity, "seed": seed, "epoch": epoch, "model": bank.state_dict(),
            "optimizer": optimizer.state_dict(), "history": history, "counts": dict(counts),
            "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if cfg["device"].startswith("cuda") else [],
            "elapsed_seconds": previous_seconds + time.perf_counter() - started,
        })
        print(f"[SUPERNET] {study.data.name} seed={seed} epoch={epoch}/{target}", flush=True)
    del optimizer
    bank.zero_grad(set_to_none=True)
    mismatch = verify_export(bank, study.panel[0], study)
    bank.to("cpu").eval()
    summary = {"epochs": target, "history": history, "activation_counts": dict(counts),
               "export_max_abs_difference": mismatch, "elapsed_seconds": previous_seconds + time.perf_counter() - started}
    atomic_json(summary_path, summary)
    return bank, summary
