"""Repeat the original CLI command after interruption; never rewrite finished fits."""
import copy
from pathlib import Path

import pytest

from mamba_tree_nas.artifacts import atomic_json, read_json
from mamba_tree_nas.cli import existing_run_action, main, run_with_reuse
from mamba_tree_nas.search import load_records, run_path, search


@pytest.mark.parametrize("phase", ["search", "refit"])
def test_empty_existing_directory_is_new(tmp_path, phase):
    tmp_path.mkdir(exist_ok=True)
    assert existing_run_action(tmp_path, phase) == "start"


@pytest.mark.parametrize("status", ["running", "interrupted", "failed", "complete"])
@pytest.mark.parametrize("phase", ["search", "refit"])
def test_existing_phase_actions(tmp_path, status, phase):
    atomic_json(tmp_path / "manifest.json", {})
    state = tmp_path / ("search_state.json" if phase == "search" else "refit/state.json")
    atomic_json(state, {"status": status})
    if phase == "search":
        atomic_json(tmp_path / "representatives.json", {})
    assert existing_run_action(tmp_path, phase) == ("skip" if status == "complete" else "resume")
    with pytest.raises(FileExistsError, match="already exists"):
        existing_run_action(tmp_path, phase, "error")


def test_manifest_only_and_incomplete_finalization_resume(tmp_path):
    atomic_json(tmp_path / "manifest.json", {})
    assert existing_run_action(tmp_path, "search") == "resume"
    atomic_json(tmp_path / "search_state.json", {"status": "complete"})
    assert existing_run_action(tmp_path, "search") == "resume"


def test_orphan_or_corrupt_state_is_not_silently_overwritten(tmp_path):
    atomic_json(tmp_path / "search_state.json", {"status": "running"})
    with pytest.raises(ValueError, match="no manifest"):
        existing_run_action(tmp_path, "search")
    atomic_json(tmp_path / "manifest.json", {})
    (tmp_path / "search_state.json").write_text("truncated JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        existing_run_action(tmp_path, "search")


def test_no_skip_claim_before_core_validation(tmp_path, capsys):
    atomic_json(tmp_path / "manifest.json", {})
    atomic_json(tmp_path / "search_state.json", {"status": "complete"})
    atomic_json(tmp_path / "representatives.json", {})
    def reject(resume):
        assert resume is True
        raise ValueError("configuration changed")
    with pytest.raises(ValueError, match="configuration changed"):
        run_with_reuse(tmp_path, "search", reject)
    assert "[SKIP]" not in capsys.readouterr().out


def test_suite_default_handles_mixed_search_and_refit_states(cfg, monkeypatch, capsys):
    import mamba_tree_nas.cli as cli
    import mamba_tree_nas.search as search_module
    import mamba_tree_nas.refit as refit_module
    cfg["data"]["primary"] = ["A", "B", "C", "D"]
    cfg["seeds"]["search"] = [2021]
    cfg["experiments"] = {"tree_direction": cfg["experiments"]["tree_direction"]}
    for dataset, status in (("A", "complete"), ("B", "interrupted"), ("D", "complete")):
        path = run_path(cfg, dataset, "tree_direction", 2021, "main")
        atomic_json(path / "manifest.json", {})
        atomic_json(path / "search_state.json", {"status": status})
        if status == "complete":
            atomic_json(path / "representatives.json", {})
        if dataset in ("A", "D"):
            atomic_json(path / "refit/state.json", {"status": "complete" if dataset == "A" else "running"})
    calls = []
    def run_search(c, dataset, variant, seed, run_id, resume, **kw):
        calls.append((dataset, "search", resume))
        return run_path(c, dataset, variant, seed, run_id)
    def run_refit(path, resume=False):
        calls.append((Path(path).parents[2].name, "refit", resume))
        return path / "refit"
    monkeypatch.setattr(cli, "load_config", lambda *a: cfg)
    monkeypatch.setattr(search_module, "search", run_search)
    monkeypatch.setattr(refit_module, "refit", run_refit)
    main(["run-suite", "--with-refit"])
    assert calls == [("A", "search", True), ("A", "refit", True), ("B", "search", True), ("B", "refit", False), ("C", "search", False), ("C", "refit", False), ("D", "search", True), ("D", "refit", True)]
    output = capsys.readouterr().out
    assert output.count("[SKIP]") == 3
    assert output.count("[RESUME]") == 2


def test_same_smoke_command_recovers_and_then_skips_without_training(cfg, monkeypatch):
    import mamba_tree_nas.cli as cli
    import mamba_tree_nas.search as search_module
    import mamba_tree_nas.refit as refit_module
    import mamba_tree_nas.report as report_module
    cfg["search"].update(population=2, budget=3)
    cfg["train"].update(epochs=1, min_epochs=1)
    cfg["calibration"].update(epochs=1, min_epochs=1)
    monkeypatch.setattr(cli, "load_config", lambda *a: cfg)
    monkeypatch.setattr(report_module, "report", lambda path, output: output)
    with pytest.raises(InterruptedError):
        search(cfg, "Synthetic", "tree_direction", 2021, synthetic=True, interrupt_after=1)
    path = run_path(cfg, "Synthetic", "tree_direction", 2021, "main")
    completed = next((path / "candidates").glob("*.json"))
    snapshot = (completed.read_bytes(), completed.stat().st_mtime_ns)
    command = ["smoke", "--dataset", "Synthetic"]
    main(command)  # No --resume required.
    assert read_json(path / "search_state.json")["status"] == "complete"
    assert read_json(path / "refit/state.json")["status"] == "complete"
    assert (completed.read_bytes(), completed.stat().st_mtime_ns) == snapshot
    files = list((path / "candidates").glob("*.json")) + list((path / "refit").rglob("seed_*.json")) + list((path / "refit").rglob("*.pt"))
    snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    def fail(*a, **kw):
        pytest.fail("Completed search/refit must not train again")
    monkeypatch.setattr(search_module, "train", fail)
    monkeypatch.setattr(refit_module, "train", fail)
    main(command)
    assert snapshots == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    with pytest.raises(FileExistsError):
        main(command + ["--existing", "error"])
    conflicting = copy.deepcopy(cfg)
    conflicting["train"]["learning_rate"] *= 2
    monkeypatch.setattr(cli, "load_config", lambda *a: conflicting)
    with pytest.raises(ValueError, match="Resume refused"):
        main(command)


def test_standalone_refit_auto_resumes_only_unfinished_seeds(cfg, monkeypatch):
    import mamba_tree_nas.cli as cli
    import mamba_tree_nas.refit as module
    cfg["search"].update(population=1, budget=1)
    cfg["train"].update(epochs=1, min_epochs=1)
    cfg["calibration"].update(epochs=1, min_epochs=1)
    path = search(cfg, "Synthetic", "tree_direction", 2021, synthetic=True)
    original_train = module.train
    def interrupt(*args, **kwargs):
        if kwargs.get("fixed_epochs") and args[3] == cfg["seeds"]["refit"][1]:
            raise KeyboardInterrupt("Interrupted between refit seeds")
        return original_train(*args, **kwargs)
    monkeypatch.setattr(module, "train", interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.refit(path)
    seed = cfg["seeds"]["refit"][0]
    first = next((path / "refit").rglob(f"seed_{seed}.json"))
    snapshot = (first.read_bytes(), first.stat().st_mtime_ns)
    calls = []
    def track(*args, **kwargs):
        assert kwargs.get("fixed_epochs") is not None, "Calibration must be reused"
        calls.append(args[3])
        return original_train(*args, **kwargs)
    monkeypatch.setattr(module, "train", track)
    monkeypatch.setattr(cli, "load_config", lambda *a: cfg)
    main(["refit", "--run", str(path)])
    assert calls == cfg["seeds"]["refit"][1:]
    assert (first.read_bytes(), first.stat().st_mtime_ns) == snapshot
    assert read_json(path / "refit/state.json")["status"] == "complete"
    calls.clear()
    main(["refit", "--run", str(path)])
    assert calls == []


def test_dry_run_previews_actions_without_execution(cfg, monkeypatch, capsys):
    import json
    import mamba_tree_nas.cli as cli
    import mamba_tree_nas.search as module
    cfg["data"]["primary"] = ["A"]
    cfg["seeds"]["search"] = [2021]
    cfg["experiments"] = {"tree_direction": cfg["experiments"]["tree_direction"]}
    monkeypatch.setattr(cli, "load_config", lambda *a: cfg)
    monkeypatch.setattr(module, "search", lambda *a, **kw: pytest.fail("dry-run launched search"))
    main(["run-suite", "--dry-run", "--with-refit"])
    result = json.loads(capsys.readouterr().out)
    assert result["run_actions"][0]["search"] == result["run_actions"][0]["refit"] == "start"
    assert not Path(cfg["output"]).exists()
