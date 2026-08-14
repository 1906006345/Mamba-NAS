from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import atomic_json, ensure_dir
from .constants import HF_DATASET_BASE, SPLIT_SEED, UEA10


@dataclass
class SeriesCollection:
    samples: list[np.ndarray]
    labels: np.ndarray
    class_names: list[str]

    @property
    def input_channels(self) -> int:
        return int(self.samples[0].shape[1])

    @property
    def max_length(self) -> int:
        return max(sample.shape[0] for sample in self.samples)


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def dataset_files(root: str | Path, dataset: str) -> tuple[Path, Path]:
    directory = Path(root) / dataset
    return directory / f"{dataset}_TRAIN.ts", directory / f"{dataset}_TEST.ts"


def validate_ts_structure(path: str | Path) -> dict[str, int]:
    """Detect truncated non-timestamp UEA files before sktime parses them."""
    path = Path(path)
    in_data = False
    timestamps = None
    expected_dimensions = None
    cases = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if not in_data:
                if lowered.startswith("@timestamps"):
                    parts = lowered.split()
                    timestamps = len(parts) > 1 and parts[1] == "true"
                elif lowered == "@data":
                    in_data = True
                continue
            if timestamps:
                # Timestamp strings may themselves contain colons; sktime remains
                # the authoritative validator for those uncommon datasets.
                cases += 1
                continue
            dimensions = line.count(":")
            if expected_dimensions is None:
                expected_dimensions = dimensions
                if expected_dimensions < 1:
                    raise OSError(f"Malformed .ts data at {path}:{line_number}: no dimensions found")
            elif dimensions != expected_dimensions:
                raise OSError(
                    f"Malformed or truncated .ts file {path}: line {line_number} has "
                    f"{dimensions} dimensions, expected {expected_dimensions}. Re-download it with "
                    f"`python -m mamba_nas.cli download --dataset {path.parent.name} --force`."
                )
            cases += 1
    if not in_data or cases == 0:
        raise OSError(f"Malformed or empty .ts file: {path}")
    return {"cases": cases, "dimensions": int(expected_dimensions or 0)}


def download_dataset(root: str | Path, dataset: str, force: bool = False) -> list[Path]:
    if dataset not in UEA10:
        raise ValueError(f"{dataset!r} is not in the uea10 suite")
    paths = dataset_files(root, dataset)
    ensure_dir(paths[0].parent)
    manifest = {"dataset": dataset, "source": HF_DATASET_BASE, "files": []}
    for split, path in zip(("TRAIN", "TEST"), paths):
        url = f"{HF_DATASET_BASE}/{dataset}/{dataset}_{split}.ts?download=true"
        if force or not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            validate_ts_structure(temporary)
            temporary.replace(path)
        else:
            validate_ts_structure(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest["files"].append(
            {"split": split, "path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    atomic_json(paths[0].parent / "source_manifest.json", manifest)
    return list(paths)


def _frame_to_samples(frame) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for _, row in frame.iterrows():
        channels = [np.asarray(cell, dtype=np.float32) for cell in row]
        length = max(len(channel) for channel in channels)
        sample = np.full((length, len(channels)), np.nan, dtype=np.float32)
        for channel_index, channel in enumerate(channels):
            sample[: len(channel), channel_index] = channel
        samples.append(_interpolate_nan(sample))
    return samples


def _interpolate_nan(sample: np.ndarray) -> np.ndarray:
    output = sample.copy()
    for channel in range(output.shape[1]):
        values = output[:, channel]
        finite = np.isfinite(values)
        if finite.any():
            positions = np.arange(len(values))
            values[~finite] = np.interp(positions[~finite], positions[finite], values[finite])
        output[:, channel] = values
    return output


def load_ts(path: str | Path, class_names: Sequence[str] | None = None) -> SeriesCollection:
    validate_ts_structure(path)
    try:
        from sktime.datasets import load_from_tsfile_to_dataframe
    except ImportError as exc:
        raise RuntimeError("sktime is required to read UEA .ts files") from exc
    frame, raw_labels = load_from_tsfile_to_dataframe(str(path), return_separate_X_and_y=True)
    raw_labels = np.asarray(raw_labels, dtype=str)
    names = list(class_names) if class_names is not None else sorted(np.unique(raw_labels).tolist())
    unknown = sorted(set(raw_labels) - set(names))
    if unknown:
        raise ValueError(f"Labels in {path} were absent from TRAIN: {unknown}")
    mapping = {name: index for index, name in enumerate(names)}
    labels = np.asarray([mapping[label] for label in raw_labels], dtype=np.int64)
    return SeriesCollection(_frame_to_samples(frame), labels, names)


def load_train(root: str | Path, dataset: str) -> SeriesCollection:
    train_path, _ = dataset_files(root, dataset)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}; run the download command first")
    return load_ts(train_path)


def load_test_for_refit(root: str | Path, dataset: str, class_names: Sequence[str]) -> SeriesCollection:
    """The only public loading entry point for TEST, intentionally named for auditability."""
    _, test_path = dataset_files(root, dataset)
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}; run the download command first")
    return load_ts(test_path, class_names)


def stratified_inner_split(labels: np.ndarray, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        rare = np.unique(labels)[counts < 2].tolist()
        raise ValueError(
            f"Stratified 80/20 split is impossible: classes {rare} have fewer than two TRAIN samples"
        )
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train, validation = train_test_split(
        indices, test_size=0.2, random_state=seed, shuffle=True, stratify=labels
    )
    return np.sort(train), np.sort(validation)


def fit_normalization(samples: Sequence[np.ndarray]) -> NormalizationStats:
    if not samples:
        raise ValueError("Cannot fit normalization without samples")
    channels = samples[0].shape[1]
    sums = np.zeros(channels, dtype=np.float64)
    squared = np.zeros(channels, dtype=np.float64)
    counts = np.zeros(channels, dtype=np.int64)
    for sample in samples:
        finite = np.isfinite(sample)
        safe = np.where(finite, sample, 0.0)
        sums += safe.sum(axis=0)
        squared += np.square(safe).sum(axis=0)
        counts += finite.sum(axis=0)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(squared, counts, out=np.ones_like(sums), where=counts > 0) - mean**2
    std = np.sqrt(np.maximum(variance, 1e-12))
    std[std < 1e-6] = 1.0
    return NormalizationStats(mean.astype(np.float32), std.astype(np.float32))


def normalize_samples(samples: Sequence[np.ndarray], stats: NormalizationStats) -> list[np.ndarray]:
    return [np.nan_to_num((sample - stats.mean) / stats.std).astype(np.float32) for sample in samples]


class PaddedSeriesDataset:
    def __init__(self, samples: Sequence[np.ndarray], labels: Sequence[int]):
        import torch

        self.samples = [torch.as_tensor(sample, dtype=torch.float32) for sample in samples]
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index], self.labels[index]


def collate_padded(batch):
    import torch

    samples, labels = zip(*batch)
    maximum = max(sample.shape[0] for sample in samples)
    channels = samples[0].shape[1]
    values = torch.zeros(len(samples), maximum, channels, dtype=torch.float32)
    mask = torch.zeros(len(samples), maximum, dtype=torch.bool)
    for index, sample in enumerate(samples):
        values[index, : sample.shape[0]] = sample
        mask[index, : sample.shape[0]] = True
    return values, mask, torch.stack(labels)


def make_loader(samples, labels, batch_size: int, shuffle: bool, seed: int):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        PaddedSeriesDataset(samples, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_padded,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def synthetic_collection(seed: int = 2021) -> SeriesCollection:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), 12)
    samples = []
    for label in labels:
        length = int(rng.integers(24, 41))
        time = np.linspace(0, 2 * np.pi, length)
        signal = np.stack(
            (np.sin((label + 1) * time), np.cos((label + 1) * time)), axis=1
        ) + rng.normal(0, 0.1, (length, 2))
        samples.append(signal.astype(np.float32))
    return SeriesCollection(samples, labels.astype(np.int64), ["class0", "class1", "class2"])
