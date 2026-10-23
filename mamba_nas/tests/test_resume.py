import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mamba_nas.config import SearchConfig
from mamba_nas.search import run_search
from mamba_nas.tests.conftest import fake_mixer_factory


def test_synthetic_search_and_resume_reuse_cache(tmp_path: Path):
    first = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=4,
        epochs=1,
        batch_size=16,
        device="cpu",
        output_dir=str(tmp_path),
    )
    dataset_dir = run_search(
        "SyntheticUEA", first, run_id="resume", synthetic=True, mixer_factory=fake_mixer_factory
    )
    lines_before = (dataset_dir / "search" / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines_before) == 4
    assert not list((dataset_dir / "search" / "candidates").glob("*/checkpoint.pt"))
    holdout_candidate = next((dataset_dir / "search" / "candidates").iterdir())
    assert not (holdout_candidate / "folds").exists()
    holdout_metrics = json.loads(
        (holdout_candidate / "metrics.json").read_text(encoding="utf-8")
    )
    assert holdout_metrics["validation_strategy"] == "holdout"
    assert holdout_metrics["cv_folds"] == 1
    second = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=8,
        epochs=1,
        batch_size=16,
        device="cpu",
        output_dir=str(tmp_path),
    )
    run_search(
        "SyntheticUEA",
        second,
        run_id="resume",
        resume=True,
        synthetic=True,
        mixer_factory=fake_mixer_factory,
    )
    lines_after = (dataset_dir / "search" / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == 8
    assert lines_after[:4] == lines_before
    assert (dataset_dir / "search" / "pareto_front.csv").exists()
    assert (dataset_dir / "search" / "search_state.pkl").exists()
    assert not list((dataset_dir / "search" / "generations").glob("generation_*_state.pkl"))
    assert not list((dataset_dir / "search" / "candidates").glob("*/checkpoint.pt"))


def test_synthetic_kfold_search_uses_shared_folds_and_oof_predictions(tmp_path: Path):
    config = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=4,
        epochs=1,
        batch_size=16,
        device="cpu",
        output_dir=str(tmp_path),
        validation_strategy="kfold",
        cv_folds=3,
    )
    dataset_dir = run_search(
        "SyntheticUEA", config, run_id="kfold", synthetic=True, mixer_factory=fake_mixer_factory
    )
    assert (dataset_dir / "cv_indices.npz").exists()
    assert not (dataset_dir / "split_indices.npz").exists()
    representatives = json.loads(
        (dataset_dir / "search" / "representatives.json").read_text(encoding="utf-8")
    )
    assert set(representatives) == {"high_f1", "knee", "low_macs"}
    candidate = next((dataset_dir / "search" / "candidates").iterdir())
    predictions = np.load(candidate / "val_predictions.npz")
    assert len(predictions["targets"]) == 36
    assert sorted(predictions["sample_indices"].tolist()) == list(range(36))
    assert len(list((candidate / "folds").glob("fold_*"))) == 3
    metrics = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation_strategy"] == "kfold"
    assert metrics["cv_folds"] == 3


def test_resume_rejects_a_different_validation_protocol(tmp_path: Path):
    holdout = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=4,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
    )
    run_search(
        "SyntheticUEA", holdout, run_id="protocol", synthetic=True, mixer_factory=fake_mixer_factory
    )
    kfold = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=8,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        validation_strategy="kfold",
        cv_folds=3,
    )
    with pytest.raises(ValueError, match="different validation protocol"):
        run_search(
            "SyntheticUEA",
            kfold,
            run_id="protocol",
            resume=True,
            synthetic=True,
            mixer_factory=fake_mixer_factory,
        )


def test_resume_rejects_a_different_search_variant(tmp_path: Path):
    baseline = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=4,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
    )
    run_search(
        "SyntheticUEA", baseline, run_id="variant", synthetic=True, mixer_factory=fake_mixer_factory
    )
    full = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=8,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        search_variant="full",
    )
    with pytest.raises(ValueError, match="different validation protocol"):
        run_search(
            "SyntheticUEA",
            full,
            run_id="variant",
            resume=True,
            synthetic=True,
            mixer_factory=fake_mixer_factory,
        )


def test_resume_rejects_different_crowding_weights(tmp_path: Path):
    first = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=4,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        search_variant="dual_crowding",
    )
    run_search(
        "SyntheticUEA", first, run_id="weights", synthetic=True, mixer_factory=fake_mixer_factory
    )
    changed = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=8,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        search_variant="dual_crowding",
        objective_weight=0.8,
        decision_weight=0.2,
    )
    with pytest.raises(ValueError, match="different validation protocol"):
        run_search(
            "SyntheticUEA",
            changed,
            run_id="weights",
            resume=True,
            synthetic=True,
            mixer_factory=fake_mixer_factory,
        )


def test_resume_rejects_revised_combo_ratio_change(tmp_path: Path):
    first = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=4,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        search_variant="archive_local_search_v2",
    )
    run_search(
        "SyntheticUEA", first, run_id="v2-ratio", synthetic=True, mixer_factory=fake_mixer_factory
    )
    changed = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=8,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        search_variant="archive_local_search_v2",
        local_search_ratio=0.10,
    )
    with pytest.raises(ValueError, match="different validation protocol"):
        run_search(
            "SyntheticUEA",
            changed,
            run_id="v2-ratio",
            resume=True,
            synthetic=True,
            mixer_factory=fake_mixer_factory,
        )


def test_revised_combo_writes_archives_slot_budget_and_cumulative_metrics(tmp_path: Path):
    config = SearchConfig.from_budget(
        "smoke",
        population_size=4,
        max_unique_candidates=8,
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        search_variant="archive_local_search_v2",
    )
    dataset_dir = run_search(
        "SyntheticUEA", config, run_id="v2-artifacts", synthetic=True,
        mixer_factory=fake_mixer_factory,
    )
    search_dir = dataset_dir / "search"
    assert (search_dir / "macro_archive.csv").exists()
    assert (search_dir / "micro_archive.csv").exists()
    budget_rows = list(csv.DictReader((search_dir / "generation_search_budget.csv").open()))
    assert budget_rows
    assert int(budget_rows[-1]["requested_local_slots"]) >= int(
        budget_rows[-1]["actual_local_children"]
    )
    diversity_rows = list(csv.DictReader((search_dir / "generation_diversity.csv").open()))
    final = diversity_rows[-1]
    for field in (
        "requested_local_slots",
        "actual_local_children",
        "fallback_local_children",
        "fallback_variation_children",
        "local_search_slot_fill_rate",
        "local_search_cumulative_positive_f1_rate",
        "local_search_cumulative_pareto_entry_rate",
        "local_search_cumulative_mean_hv_contribution",
    ):
        assert field in final
