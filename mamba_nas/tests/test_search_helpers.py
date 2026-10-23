import json
from types import SimpleNamespace

import numpy as np

from mamba_nas.config import SearchConfig
from mamba_nas.search import (
    CandidateEvaluator,
    DiversityArchive,
    MacroMicroArchive,
    architecture_distance,
    decision_crowding,
    coupled_scan_neighbors,
    nondominated,
    one_gene_neighbors,
    select_representatives,
    _coupled_local_search_vectors,
)
from mamba_nas.search_space import Genome
from mamba_nas.refit import canonical_representatives


def test_pareto_and_representative_selection():
    rows = [
        {"candidate_hash": "a", "macro_f1": 0.9, "macs": 100, "objective_f1": 0.1, "objective_macs": 4.6},
        {"candidate_hash": "b", "macro_f1": 0.8, "macs": 50, "objective_f1": 0.2, "objective_macs": 3.9},
        {"candidate_hash": "c", "macro_f1": 0.7, "macs": 150, "objective_f1": 0.3, "objective_macs": 5.0},
    ]
    front = nondominated(rows)
    assert {row["candidate_hash"] for row in front} == {"a", "b"}
    selected = select_representatives(front)
    assert selected["high_f1"]["candidate_hash"] == "a"
    assert selected["low_macs"]["candidate_hash"] == "b"


def test_restart_cache_reuses_completed_but_retries_failed(tmp_path):
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    rows = [
        {"candidate_hash": "failed", "status": "failed"},
        {"candidate_hash": "completed", "status": "completed", "macro_f1": 0.8},
    ]
    (search_dir / "evaluations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    evaluator = CandidateEvaluator.__new__(CandidateEvaluator)
    evaluator.search_dir = search_dir
    assert evaluator._load_cache() == {"completed": rows[1]}


def test_legacy_representative_names_are_read_with_canonical_roles():
    representatives = canonical_representatives(
        {
            "high_accuracy": {"candidate_hash": "a"},
            "knee": {"candidate_hash": "b"},
            "low_cost": {"candidate_hash": "c"},
        }
    )
    assert set(representatives) == {"high_f1", "knee", "low_macs"}


def _genome(**overrides):
    values = {
        "tokenizer": "point",
        "num_blocks": 1,
        "direction": "forward",
        "d_model": 64,
        "d_state": 8,
        "d_conv": 2,
        "expand": 1,
        "pooling": "mean",
    }
    values.update(overrides)
    return Genome(**values)


def _row(genome, candidate_hash, f1, macs):
    return {
        "candidate_hash": candidate_hash,
        **genome.to_dict(),
        "macro_f1": f1,
        "macs": macs,
        "objective_f1": 1.0 - f1,
        "objective_macs": float(macs),
        "status": "completed",
    }


def test_architecture_distance_uses_categorical_values_and_weights():
    base = _genome()
    assert architecture_distance(base, base) == 0
    assert architecture_distance(base, _genome(tokenizer="patch8")) == 2.0
    assert architecture_distance(base, _genome(direction="bidirectional")) == 2.0
    assert architecture_distance(base, _genome(d_state=16)) == 0.5


def test_one_gene_neighbors_are_legal_and_change_exactly_one_gene():
    base = _genome()
    neighbors = one_gene_neighbors(base)
    assert len(neighbors) == 11
    for child, changed in neighbors:
        child.validate()
        differences = [name for name in base.to_dict() if getattr(base, name) != getattr(child, name)]
        assert differences == [changed]


def test_diversity_archive_keeps_local_pareto_elites_per_niche():
    archive = DiversityArchive(max_elites_per_niche=3)
    genome = _genome()
    for index, (f1, macs) in enumerate(((0.7, 100), (0.8, 120), (0.9, 140), (0.85, 110))):
        archive.update(_row(genome, str(index), f1, macs), generation=index)
    assert len(archive.rows()) <= 3
    assert len({row["candidate_hash"] for row in archive.rows()}) == len(archive.rows())


def test_feature_only_config_overrides_create_custom_variant():
    config = SearchConfig.from_budget("smoke", use_decision_crowding=True)
    assert config.search_variant == "custom"
    assert config.validate().use_decision_crowding


def test_archive_local_search_variant_enables_only_archive_and_local_search():
    config = SearchConfig.from_budget("smoke", search_variant="archive_local_search")
    config.validate()
    assert not config.use_decision_crowding
    assert config.use_diversity_archive
    assert config.use_pareto_local_search


def test_crowding_weights_are_configurable_and_must_sum_to_one():
    config = SearchConfig.from_budget(
        "smoke",
        search_variant="dual_crowding",
        objective_weight=0.8,
        decision_weight=0.2,
    )
    assert config.validate().objective_weight == 0.8
    invalid = SearchConfig.from_budget(
        "smoke",
        search_variant="dual_crowding",
        objective_weight=0.8,
        decision_weight=0.3,
    )
    import pytest

    with pytest.raises(ValueError, match="must equal 1.0"):
        invalid.validate()


def test_macro_micro_archive_has_one_macro_and_at_most_three_micro_elites():
    archive = MacroMicroArchive(max_elites_per_niche=3)
    genome = _genome()
    for index, (f1, macs) in enumerate(((0.7, 100), (0.8, 120), (0.9, 140), (0.85, 110))):
        archive.update(_row(genome, str(index), f1, macs), generation=index)
    key = next(iter(archive.entries))
    assert len(archive.macro_rows()) == 1
    assert len(archive.micro_rows()) <= 3
    assert archive.macro_rows()[0]["candidate_hash"] in {
        row["candidate_hash"] for row in archive.micro_rows()
    }


def test_revised_archive_local_search_defaults_to_layered_ratios():
    config = SearchConfig.from_budget("smoke", search_variant="archive_local_search_v2")
    config.validate()
    assert config.use_macro_micro_archive
    assert config.local_search_ratio == 0.20
    assert (config.archive_seed_ratio, config.pareto_seed_ratio, config.population_seed_ratio) == (0.30, 0.50, 0.20)


def test_search_algorithm_choices_and_variant_compatibility():
    assert SearchConfig.from_budget("smoke", search_algorithm="random").validate().search_algorithm == "random"
    assert SearchConfig.from_budget("smoke", search_algorithm="moead").validate().search_algorithm == "moead"
    import pytest

    with pytest.raises(ValueError, match="only supported"):
        SearchConfig.from_budget(
            "smoke", search_algorithm="moead", search_variant="archive_local_search"
        ).validate()


def test_scan_order_variants_enable_expected_features():
    scan = SearchConfig.from_budget("smoke", search_variant="scan_order").validate()
    local = SearchConfig.from_budget(
        "smoke", search_variant="scan_order_local_search"
    ).validate()
    coupled = SearchConfig.from_budget(
        "smoke", search_variant="scan_order_coupled_local_search"
    ).validate()
    dual = SearchConfig.from_budget(
        "smoke", search_variant="scan_order_dual_crowding"
    ).validate()
    assert scan.use_scan_order and not scan.use_pareto_local_search
    assert local.use_scan_order and local.use_pareto_local_search
    assert not local.use_coupled_local_search and local.local_search_ratio == 0.20
    assert coupled.use_coupled_local_search
    assert coupled.coupled_neighbor_ratio == 0.30
    assert not coupled.use_diversity_archive
    assert dual.use_scan_order and dual.use_decision_crowding
    assert not dual.use_diversity_archive and not dual.use_pareto_local_search


def test_coupled_scan_neighbors_change_exactly_scan_and_one_partner():
    parent = _genome(tokenizer="patch8", scan_order="time_major")
    neighbors = coupled_scan_neighbors(parent)
    assert neighbors
    for child, changed, partner in neighbors:
        child.validate()
        differences = {
            name for name in parent.to_dict()
            if getattr(parent, name) != getattr(child, name)
        }
        assert differences == {"scan_order", partner}
        assert partner in {"tokenizer", "direction", "d_conv"}
        assert changed == f"scan_order+{partner}"


def test_point_can_enter_interleaved_only_via_coupled_tokenizer_change():
    parent = _genome(tokenizer="point", scan_order="time_major")
    single = one_gene_neighbors(parent, use_scan_order=True)
    assert not any(child.scan_order == "patch_interleaved" for child, _ in single)
    coupled = coupled_scan_neighbors(parent)
    assert {child.tokenizer for child, _, _ in coupled} == {"patch8", "patch16"}


def test_coupled_local_search_uses_seventy_thirty_split_without_archive():
    parent = _genome(tokenizer="patch8", scan_order="time_major")
    row = _row(parent, parent.sha256, 0.8, 100)
    evaluator = SimpleNamespace(
        cache={parent.sha256: row},
        config=SimpleNamespace(
            coupled_neighbor_ratio=0.30,
            local_search_max_neighbors=12,
        ),
    )
    records = []
    vectors = _coupled_local_search_vectors(
        evaluator, [], 10, np.random.default_rng(2021), 2, records, set()
    )
    assert len(vectors) == len(records) == 10
    assert sum(row["neighbor_type"] == "single" for row in records) == 7
    assert sum(row["neighbor_type"] == "coupled" for row in records) == 3


def test_fractional_coupled_quota_is_thirty_percent_across_generations():
    parent = _genome(tokenizer="patch8", scan_order="time_major")
    row = _row(parent, parent.sha256, 0.8, 100)
    coupled_counts = []
    for generation in range(1, 5):
        evaluator = SimpleNamespace(
            cache={parent.sha256: row},
            config=SimpleNamespace(
                coupled_neighbor_ratio=0.30,
                local_search_max_neighbors=12,
            ),
        )
        records = []
        _coupled_local_search_vectors(
            evaluator, [], 5, np.random.default_rng(2021), generation, records, set()
        )
        coupled_counts.append(
            sum(item["neighbor_type"] == "coupled" for item in records)
        )
    assert coupled_counts == [2, 1, 2, 1]
    assert sum(coupled_counts) / 20 == 0.30
