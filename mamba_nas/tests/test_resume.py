from pathlib import Path

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
