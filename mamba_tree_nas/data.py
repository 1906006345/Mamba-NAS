from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from .config import digest, resolve_path


def read_ts(path):
    """UEA .ts reader, including unequal lengths and numeric timestamp tuples.

    Class labels are strings. Timestamp values are ordering annotations only;
    irregular timestamp interpolation is not silently invented.
    """
    samples, labels = [], []
    in_data = False
    timestamps = False
    class_labels = False
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if not in_data:
                if lower.startswith("@timestamps"):
                    timestamps = lower.split()[1] == "true"
                if lower.startswith("@classlabel"):
                    class_labels = lower.split()[1] == "true"
                if lower == "@data":
                    in_data = True
                continue
            if not class_labels:
                raise ValueError("A labelled classification .ts file is required")
            # Split dimension separators outside timestamp parentheses.
            pieces, start, depth = [], 0, 0
            for i, char in enumerate(line):
                depth += (char == "(") - (char == ")")
                if char == ":" and depth == 0:
                    pieces.append(line[start:i]); start = i + 1
            pieces.append(line[start:])
            channels = []
            for part in pieces[:-1]:
                if timestamps:
                    import re
                    pairs = re.findall(r"\(([^()]*)\)", part)
                    values = [pair.rsplit(",", 1)[1] for pair in pairs]
                    if not values:
                        raise ValueError(f"Empty timestamp dimension on line {number}")
                else:
                    values = part.split(",")
                channels.append(np.array([float(v) if v.strip() not in ("?", "", "NaN", "nan") else np.nan for v in values], dtype=np.float64))
            if not channels or len({len(x) for x in channels}) != 1:
                raise ValueError(f"Empty or asynchronous dimensions at {path}:{number}; explicit alignment is required")
            sample = np.stack(channels, axis=1)
            if sample.shape[0] == 0:
                raise ValueError("Empty sequence")
            samples.append(interpolate_sample(sample)); labels.append(pieces[-1].strip())
    if not samples or not in_data or len({x.shape[1] for x in samples}) != 1:
        raise ValueError(f"Invalid classification dataset: {path}")
    return samples, labels


def interpolate_sample(sample):
    sample = np.asarray(sample, dtype=np.float64).copy()
    index = np.arange(len(sample))
    for c in range(sample.shape[1]):
        valid = np.isfinite(sample[:, c])
        sample[:, c] = np.interp(index, index[valid], sample[valid, c]) if valid.any() else 0.0
    return sample.astype(np.float32)


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, samples, epsilon):
        # Streaming sums: no padded/validation/TEST entries, no giant concatenate.
        count, total, square = 0, None, None
        for x in samples:
            y = x.astype(np.float64)
            count += len(y)
            total = y.sum(0) if total is None else total + y.sum(0)
            square = (y * y).sum(0) if square is None else square + (y * y).sum(0)
        mean = total / count
        return cls(mean, np.sqrt(np.maximum(square / count - mean * mean, epsilon * epsilon)))

    def transform(self, samples):
        return [((x - self.mean) / self.scale).astype(np.float32) for x in samples]

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "fit": "unmasked training timepoints per channel"}


def stratified_split(labels, fraction, seed):
    from sklearn.model_selection import train_test_split
    labels = np.asarray(labels)
    counts = np.unique(labels, return_counts=True)[1]
    nval = math.ceil(len(labels) * fraction)
    if min(counts) < 2 or min(nval, len(labels) - nval) < len(counts):
        raise ValueError("Stratified 80/20 holdout is infeasible for rare classes; protocol was not changed")
    train, val = train_test_split(np.arange(len(labels)), test_size=fraction, stratify=labels, random_state=seed)
    return sorted(train.tolist()), sorted(val.tolist())


def balanced_indices(labels, cfg, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in np.unique(labels)]
    size = min(len(labels), max(cfg["minimum_batch"], cfg["examples_per_class"] * len(groups)))
    selected = []
    while len(selected) < size:
        for group in groups:
            if group and len(selected) < size:
                selected.append(group.pop())
    return selected


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, indices=None):
        self.samples = samples
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = list(range(len(labels))) if indices is None else list(indices)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(self.samples[i]), int(self.labels[i]), self.indices[i]


def collate(batch):
    lengths = [len(x[0]) for x in batch]
    x = torch.zeros(len(batch), max(lengths), batch[0][0].shape[1])
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    for i, (sample, _, _) in enumerate(batch):
        x[i, :len(sample)] = sample
    return x, mask, torch.tensor([x[1] for x in batch]), torch.tensor([x[2] for x in batch])


class DataProtocol:
    def __init__(self, cfg, name, synthetic=False):
        self.cfg, self.name, self.synthetic = cfg, name, synthetic
        if synthetic:
            # No TEST is generated or accessed until load_test().
            self.samples, labels = self._synthetic(False)
            self.source_hash = digest({"synthetic": cfg["smoke"], "seed": cfg["data"]["split_seed"]})
        else:
            path = resolve_path(cfg["data"]["root"]) / name / f"{name}_TRAIN.ts"
            self.samples, labels = read_ts(path)
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            self.source_hash = hasher.hexdigest()
        self.classes = sorted(set(labels))
        self.mapping = {x: i for i, x in enumerate(self.classes)}
        self.labels = np.array([self.mapping[x] for x in labels])
        self.train_indices, self.val_indices = stratified_split(self.labels, cfg["data"]["validation_fraction"], cfg["data"]["split_seed"])
        self.normalizer = Normalizer.fit([self.samples[i] for i in self.train_indices], cfg["data"]["normalization_epsilon"])
        norm = self.normalizer.transform(self.samples)
        self.train = SequenceDataset([norm[i] for i in self.train_indices], self.labels[self.train_indices], self.train_indices)
        self.val = SequenceDataset([norm[i] for i in self.val_indices], self.labels[self.val_indices], self.val_indices)
        self.channels = self.samples[0].shape[1]
        self.max_length = max(map(len, self.samples))
        self.proxy_indices = balanced_indices(self.train.labels, cfg["proxy"], cfg["data"]["split_seed"])
        self.manifest = {"dataset": name, "source_train_sha256": self.source_hash, "train_indices": self.train_indices, "validation_indices": self.val_indices, "split_seed": cfg["data"]["split_seed"], "class_mapping": self.mapping, "normalization": self.normalizer.to_dict(), "missing_values": "per-sample per-channel linear interpolation; endpoint carry; all missing -> zero", "proxy_indices_in_train": self.proxy_indices, "proxy_source_indices": [self.train_indices[i] for i in self.proxy_indices], "channels": self.channels, "max_train_length": self.max_length, "test_accessed": False, "synthetic": synthetic}
        self.fingerprint = digest(self.manifest)

    def _synthetic(self, test):
        c = self.cfg["smoke"]
        rng = np.random.default_rng(self.cfg["data"]["split_seed"] + (1 if test else 0))
        count = c["test_samples"] if test else c["samples"]
        labels = [str(i % c["classes"]) for i in range(count)]
        return [interpolate_sample(rng.normal(size=(c["length"] - i % 4, c["channels"])) + int(label) * .3) for i, label in enumerate(labels)], labels

    def full_train(self):
        normalizer = Normalizer.fit(self.samples, self.cfg["data"]["normalization_epsilon"])
        return SequenceDataset(normalizer.transform(self.samples), self.labels), normalizer

    def load_test(self, normalizer):
        if self.synthetic:
            samples, labels = self._synthetic(True)
        else:
            samples, labels = read_ts(resolve_path(self.cfg["data"]["root"]) / self.name / f"{self.name}_TEST.ts")
        if any(label not in self.mapping for label in labels) or any(x.shape[1] != self.channels for x in samples):
            raise ValueError("TEST class/channel schema differs from TRAIN")
        return SequenceDataset(normalizer.transform(samples), [self.mapping[x] for x in labels])
