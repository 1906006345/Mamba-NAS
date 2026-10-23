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
    validation_strategy: str = "holdout"
    cv_folds: int = 5
    search_algorithm: str = "nsga2"
    search_variant: str = "baseline"
    use_decision_crowding: bool = False
    use_diversity_archive: bool = False
    use_pareto_local_search: bool = False
    use_macro_micro_archive: bool = False
    use_scan_order: bool = False
    use_coupled_local_search: bool = False
    hypervolume_space: str = "active"
    objective_weight: float = 0.70
    decision_weight: float = 0.30
    diversity_parent_ratio: float = 0.30
    random_immigrant_ratio: float = 0.10
    local_search_ratio: float = 0.30
    local_search_max_neighbors: int = 12
    coupled_neighbor_ratio: float = 0.30
    archive_seed_ratio: float = 0.30
    pareto_seed_ratio: float = 0.50
    population_seed_ratio: float = 0.20

    @classmethod
    def from_budget(cls, budget: str, **overrides: Any) -> "SearchConfig":
        if budget not in BUDGETS:
            raise ValueError(f"Unknown budget {budget!r}; choose from {tuple(BUDGETS)}")
        values = {"budget": budget, **BUDGETS[budget], **overrides}
        variant_flags = {
            "baseline": (False, False, False, False, False),
            "dual_crowding": (True, False, False, False, False),
            "diversity_archive": (False, True, False, False, False),
            "dual_space": (True, True, False, False, False),
            "local_search": (False, False, True, False, False),
            "archive_local_search": (False, True, True, False, False),
            "archive_local_search_v2": (False, True, True, False, False),
            "full": (True, True, True, False, False),
            "scan_order": (False, False, False, True, False),
            "scan_order_local_search": (False, False, True, True, False),
            "scan_order_coupled_local_search": (False, False, True, True, True),
            "scan_order_dual_crowding": (True, False, False, True, False),
        }
        requested_variant = overrides.get("search_variant")
        if requested_variant in variant_flags:
            names = (
                "use_decision_crowding",
                "use_diversity_archive",
                "use_pareto_local_search",
                "use_scan_order",
                "use_coupled_local_search",
            )
            for name, enabled in zip(names, variant_flags[requested_variant]):
                values.setdefault(name, enabled)
            if requested_variant == "archive_local_search_v2":
                values.setdefault("use_macro_micro_archive", True)
                values.setdefault("local_search_ratio", 0.20)
            if requested_variant in {
                "scan_order_local_search", "scan_order_coupled_local_search"
            }:
                values.setdefault("local_search_ratio", 0.20)
        # Feature-only overrides form an explicit custom ablation. Named
        # variants remain strict when search_variant is supplied by the caller.
        feature_names = {
            "use_decision_crowding",
            "use_diversity_archive",
            "use_pareto_local_search",
            "use_macro_micro_archive",
            "use_scan_order",
            "use_coupled_local_search",
        }
        if "search_variant" not in overrides and feature_names.intersection(overrides):
            values["search_variant"] = "custom"
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "SearchConfig":
        if self.validation_strategy not in {"holdout", "kfold"}:
            raise ValueError(
                "validation_strategy must be either 'holdout' or 'kfold', got "
                f"{self.validation_strategy!r}"
            )
        if self.search_algorithm not in {"random", "nsga2", "moead"}:
            raise ValueError(
                "search_algorithm must be one of ('random', 'nsga2', 'moead'), got "
                f"{self.search_algorithm!r}"
            )
        if self.hypervolume_space not in {"active", "scan_order"}:
            raise ValueError("hypervolume_space must be 'active' or 'scan_order'")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be at least 2, got {self.cv_folds}")
        variants = {
            "custom": None,
            "baseline": (False, False, False, False, False),
            "dual_crowding": (True, False, False, False, False),
            "diversity_archive": (False, True, False, False, False),
            "dual_space": (True, True, False, False, False),
            "local_search": (False, False, True, False, False),
            "archive_local_search": (False, True, True, False, False),
            "archive_local_search_v2": (False, True, True, False, False),
            "full": (True, True, True, False, False),
            "scan_order": (False, False, False, True, False),
            "scan_order_local_search": (False, False, True, True, False),
            "scan_order_coupled_local_search": (False, False, True, True, True),
            "scan_order_dual_crowding": (True, False, False, True, False),
        }
        if self.search_variant not in variants:
            raise ValueError(
                f"search_variant must be one of {tuple(variants)}, got {self.search_variant!r}"
            )
        expected = variants[self.search_variant]
        actual = (
            self.use_decision_crowding,
            self.use_diversity_archive,
            self.use_pareto_local_search,
            self.use_scan_order,
            self.use_coupled_local_search,
        )
        if expected is not None and actual != expected:
            raise ValueError(
                f"search_variant={self.search_variant!r} conflicts with feature flags {actual}; "
                f"expected {expected}"
            )
        if self.search_algorithm != "nsga2" and self.search_variant != "baseline":
            raise ValueError(
                "search_variant mechanisms are only supported with search_algorithm='nsga2'"
            )
        if self.search_variant == "archive_local_search_v2" and not self.use_macro_micro_archive:
            raise ValueError("archive_local_search_v2 requires use_macro_micro_archive=True")
        if self.use_coupled_local_search and not (
            self.use_scan_order and self.use_pareto_local_search
        ):
            raise ValueError(
                "coupled local search requires scan order and Pareto local search"
            )
        for name in ("objective_weight", "decision_weight"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if abs(self.objective_weight + self.decision_weight - 1.0) > 1e-9:
            raise ValueError(
                "objective_weight + decision_weight must equal 1.0, got "
                f"{self.objective_weight + self.decision_weight}"
            )
        for name in (
            "diversity_parent_ratio",
            "random_immigrant_ratio",
            "local_search_ratio",
            "archive_seed_ratio",
            "pareto_seed_ratio",
            "population_seed_ratio",
            "coupled_neighbor_ratio",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.diversity_parent_ratio + self.random_immigrant_ratio > 1:
            raise ValueError("diversity_parent_ratio + random_immigrant_ratio must be <= 1")
        if abs(
            self.archive_seed_ratio
            + self.pareto_seed_ratio
            + self.population_seed_ratio
            - 1.0
        ) > 1e-9:
            raise ValueError("archive_seed_ratio + pareto_seed_ratio + population_seed_ratio must equal 1.0")
        if self.local_search_max_neighbors < 1:
            raise ValueError("local_search_max_neighbors must be at least 1")
        return self

    def resolved_output(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()
