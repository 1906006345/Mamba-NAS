from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

GENE_NAMES = (
    "tokenizer",
    "num_blocks",
    "direction",
    "d_model",
    "d_state",
    "d_conv",
    "expand",
    "pooling",
)
SCAN_GENE_NAMES = (*GENE_NAMES, "scan_order")

CHOICES = {
    "tokenizer": ("point", "patch8", "patch16"),
    "num_blocks": (1, 2, 3),
    "direction": ("forward", "bidirectional"),
    "d_model": (64, 128),
    "d_state": (8, 16),
    "d_conv": (2, 4),
    "expand": (1, 2),
    "pooling": ("mean", "max", "meanmax"),
    "scan_order": ("time_major", "patch_interleaved"),
}


@dataclass(frozen=True)
class Genome:
    tokenizer: str
    num_blocks: int
    direction: str
    d_model: int
    d_state: int
    d_conv: int
    expand: int
    pooling: str
    scan_order: str = "time_major"

    def validate(self) -> "Genome":
        for name in SCAN_GENE_NAMES:
            value = getattr(self, name)
            if value not in CHOICES[name]:
                raise ValueError(f"Invalid {name}={value!r}; expected one of {CHOICES[name]}")
        if self.scan_order == "patch_interleaved" and self.tokenizer == "point":
            raise ValueError("patch_interleaved requires tokenizer='patch8' or 'patch16'")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        values = self.to_dict()
        # Preserve hashes of every legacy architecture and its completed cache.
        if self.scan_order == "time_major":
            values.pop("scan_order")
        return json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def active_gene_names(use_scan_order: bool = False) -> tuple[str, ...]:
    return SCAN_GENE_NAMES if use_scan_order else GENE_NAMES


def bounds(use_scan_order: bool = False) -> tuple[list[int], list[int]]:
    names = active_gene_names(use_scan_order)
    return [0] * len(names), [len(CHOICES[n]) - 1 for n in names]


def decode(vector: Sequence[int]) -> Genome:
    if len(vector) not in {len(GENE_NAMES), len(SCAN_GENE_NAMES)}:
        raise ValueError(
            f"Expected {len(GENE_NAMES)} legacy genes or {len(SCAN_GENE_NAMES)} "
            f"scan-order genes, got {len(vector)}"
        )
    names = SCAN_GENE_NAMES if len(vector) == len(SCAN_GENE_NAMES) else GENE_NAMES
    values = {}
    for index, name in enumerate(names):
        gene = int(vector[index])
        if gene < 0 or gene >= len(CHOICES[name]):
            raise ValueError(f"Gene {name} index {gene} is out of bounds")
        values[name] = CHOICES[name][gene]
    return Genome(**values).validate()


def encode(genome: Genome, use_scan_order: bool = False) -> tuple[int, ...]:
    genome.validate()
    if genome.scan_order != "time_major" and not use_scan_order:
        raise ValueError("Encoding patch_interleaved requires use_scan_order=True")
    names = active_gene_names(use_scan_order)
    return tuple(CHOICES[name].index(getattr(genome, name)) for name in names)


def enumerate_space(use_scan_order: bool = False) -> Iterable[Genome]:
    names = active_gene_names(use_scan_order)
    for values in itertools.product(*(CHOICES[name] for name in names)):
        try:
            yield Genome(**dict(zip(names, values))).validate()
        except ValueError:
            continue


def space_size(use_scan_order: bool = False) -> int:
    return sum(1 for _ in enumerate_space(use_scan_order))


def genome_from_mapping(values: dict) -> Genome:
    """Read current or legacy rows, defaulting absent scan_order to time_major."""
    return Genome(
        **{name: values[name] for name in GENE_NAMES},
        scan_order=values.get("scan_order", "time_major") or "time_major",
    ).validate()


def repair_vector(
    vector: Sequence[int],
    original: Sequence[int] | None = None,
    rng=None,
) -> tuple[int, ...]:
    """Repair the only conditional constraint in the scan-order space."""
    result = [int(value) for value in vector]
    if len(result) != len(SCAN_GENE_NAMES):
        return tuple(result)
    tokenizer_index = SCAN_GENE_NAMES.index("tokenizer")
    scan_index = SCAN_GENE_NAMES.index("scan_order")
    point = CHOICES["tokenizer"].index("point")
    interleaved = CHOICES["scan_order"].index("patch_interleaved")
    if result[tokenizer_index] != point or result[scan_index] != interleaved:
        return tuple(result)
    original_values = list(original) if original is not None else None
    tokenizer_changed = bool(
        original_values is not None
        and result[tokenizer_index] != int(original_values[tokenizer_index])
    )
    scan_changed = bool(
        original_values is not None
        and result[scan_index] != int(original_values[scan_index])
    )
    if tokenizer_changed and not scan_changed:
        result[scan_index] = CHOICES["scan_order"].index("time_major")
    else:
        patch_indices = [
            CHOICES["tokenizer"].index("patch8"),
            CHOICES["tokenizer"].index("patch16"),
        ]
        result[tokenizer_index] = int(
            rng.choice(patch_indices) if rng is not None else patch_indices[0]
        )
    return tuple(result)

