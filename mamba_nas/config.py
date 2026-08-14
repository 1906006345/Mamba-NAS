from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import BUDGETS, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class SearchConfig:
    budget: str
    population_size: int
    max_unique_candidates: int
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int = 2021
    data_dir: str = DEFAULT_DATA_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "cuda"
    dropout: float = 0.1

    @classmethod
    def from_budget(cls, budget: str, **overrides: Any) -> "SearchConfig":
        if budget not in BUDGETS:
            raise ValueError(f"Unknown budget {budget!r}; choose from {tuple(BUDGETS)}")
        values = {"budget": budget, **BUDGETS[budget], **overrides}
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    def resolved_output(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

