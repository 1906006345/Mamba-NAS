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

CHOICES = {
    "tokenizer": ("point", "patch8", "patch16"),
    "num_blocks": (1, 2, 3),
    "direction": ("forward", "bidirectional"),
    "d_model": (64, 128),
    "d_state": (8, 16),
    "d_conv": (2, 4),
    "expand": (1, 2),
    "pooling": ("mean", "max", "meanmax"),
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

    def validate(self) -> "Genome":
        for name in GENE_NAMES:
            value = getattr(self, name)
            if value not in CHOICES[name]:
                raise ValueError(f"Invalid {name}={value!r}; expected one of {CHOICES[name]}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def bounds() -> tuple[list[int], list[int]]:
    return [0] * len(GENE_NAMES), [len(CHOICES[n]) - 1 for n in GENE_NAMES]


def decode(vector: Sequence[int]) -> Genome:
    if len(vector) != len(GENE_NAMES):
        raise ValueError(f"Expected {len(GENE_NAMES)} genes, got {len(vector)}")
    values = {}
    for index, name in enumerate(GENE_NAMES):
        gene = int(vector[index])
        if gene < 0 or gene >= len(CHOICES[name]):
            raise ValueError(f"Gene {name} index {gene} is out of bounds")
        values[name] = CHOICES[name][gene]
    return Genome(**values).validate()


def encode(genome: Genome) -> tuple[int, ...]:
    genome.validate()
    return tuple(CHOICES[name].index(getattr(genome, name)) for name in GENE_NAMES)


def enumerate_space() -> Iterable[Genome]:
    for values in itertools.product(*(CHOICES[name] for name in GENE_NAMES)):
        yield Genome(**dict(zip(GENE_NAMES, values)))


def space_size() -> int:
    size = 1
    for name in GENE_NAMES:
        size *= len(CHOICES[name])
    return size

