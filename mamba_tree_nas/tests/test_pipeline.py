import copy
import json
import random
from pathlib import Path
import pytest
from mamba_tree_nas.artifacts import read_json
from mamba_tree_nas.cli import budget_summary, matrix, parser, verify
from mamba_tree_nas.config import load_config
from mamba_tree_nas.refit import refit
from mamba_tree_nas.report import report
from mamba_tree_nas.search import load_records, search


def test_matrix_budget():
    cfg = load_config()
    args = parser().parse_args(["run-suite", "--dry-run"])
    runs = matrix(cfg, args)
    summary = budget_summary(cfg, runs, "main", True)
    assert summary["search_runs"] == 45
    assert summary["candidate_evaluations_upper"] == 17280
    assert summary["calibrations_upper_if_refit"] == 135
    assert summary["refit_fits_upper_if_refit"] == 405
    args.suite = "uea10"
    assert len(matrix(cfg, args)) == 150


def test_suite_resume_starts_not_yet_created_runs(cfg, monkeypatch):
    import mamba_tree_nas.cli as cli
    import mamba_tree_nas.search as module
    cfg["data"]["primary"] = ["A", "B"]
    cfg["seeds"]["search"] = [2021]
    cfg["experiments"] = {"tree_direction":cfg["experiments"]["tree_direction"]}
    first = module.run_path(cfg, "A", "tree_direction", 2021, "main")
    first.mkdir(parents=True)
    (first / "manifest.json").write_text("{}")
    calls = []
    def run(c, dataset, variant, seed, run_id, resume, **kw):
        calls.append((dataset, resume))
        return module.run_path(c, dataset, variant, seed, run_id)
    monkeypatch.setattr(cli, "load_config", lambda *a: cfg)
    monkeypatch.setattr(module, "search", run)
    cli.main(["run-suite", "--resume"])
    assert calls == [("A", True), ("B", False)]


def test_cpu_production_rejected(cfg):
    cfg["backend"] = "mamba"
    import torch
    from mamba_tree_nas.model import mixer_factory
    with pytest.raises(RuntimeError, match="Production"):
        mixer_factory("mamba", "cpu")
    cfg["backend"] = "test"
    with pytest.raises(ValueError, match="synthetic"):
        search(cfg, "A", "tree_direction", 2021)


def test_full_cpu_pipeline_resume_retention_report(cfg):
    with pytest.raises(InterruptedError):
        search(cfg, "Synthetic", "tree_direction", 2021, "interrupted", synthetic=True, interrupt_after=2)
    path = search(cfg, "Synthetic", "tree_direction", 2021, "interrupted", resume=True, synthetic=True)
    records = load_records(path)
    assert len(records) == cfg["search"]["budget"] and all(r["status"] == "complete" for r in records)
    assert all(r["proxy"]["status"] == "complete" for r in records)
    clean = search(cfg, "Synthetic", "tree_direction", 2021, "clean", synthetic=True)
    clean_records = load_records(clean)
    assert [r["hash"] for r in records] == [r["hash"] for r in clean_records]
    assert [r["validation"] for r in records] == [r["validation"] for r in clean_records]
    assert search(cfg, "Synthetic", "tree_direction", 2021, "interrupted", resume=True, synthetic=True) == path
    changed = copy.deepcopy(cfg); changed["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="Resume refused"):
        search(changed, "Synthetic", "tree_direction", 2021, "interrupted", resume=True, synthetic=True)
    assert not list((path / "candidates").rglob("*.pt"))
    refit(path)
    assert read_json(path / "refit" / "state.json")["status"] == "complete"
    chosen = read_json(path / "representatives.json")
    unique = len(chosen["architectures"])
    assert len(list((path / "refit").rglob("*.pt"))) == unique
    assert len(list((path / "refit").rglob("*_predictions.npz"))) == unique * 3
    assert len(read_json(path / "refit" / "state.json")["completed"]) == unique * 3
    assert refit(path, resume=True) == path / "refit"
    export = Path(cfg["exports"])
    report(path, export)
    assert (export / "results.csv").exists()
    assert list(export.rglob("*.pdf")) and list(export.rglob("*.png"))
    assert "SYNTHETIC" in read_json(export / "report_manifest.json")["synthetic_warning"]
    import numpy as np
    for file in (path / "refit").rglob("*_predictions.npz"):
        with np.load(file) as p:
            assert set(p.files) == {"sample_index", "label", "prediction", "probabilities"}
            assert np.allclose(p["probabilities"].sum(1), 1)
    from mamba_tree_nas.inference import load_retained_model
    from mamba_tree_nas.data import DataProtocol
    from mamba_tree_nas.training import evaluate
    weight_file = next((path / "refit").rglob("first_seed.pt"))
    model, normalizer, classes = load_retained_model(weight_file, "cpu")
    data = DataProtocol(cfg, "Synthetic", True)
    metrics, predictions = evaluate(model, data.load_test(normalizer), cfg, len(classes))
    saved = read_json(weight_file.parent / f"seed_{cfg['seeds']['refit'][0]}.json")
    assert metrics == saved["test"]
    with pytest.raises(ValueError, match="Duplicate"):
        # Avoid rerendering irrelevant figures in this guard test.
        import mamba_tree_nas.report as module
        from unittest.mock import patch
        with patch.object(module, "_run_figures"):
            refit(clean)
            report(Path(cfg["output"]), export / "duplicate")


def test_proxy_failure_is_not_training_failure(cfg, monkeypatch):
    import mamba_tree_nas.search as module
    monkeypatch.setattr(module, "measure_proxy", lambda *a: {"status":"failed", "seconds":0, "error":{"message":"proxy-only failure"}})
    cfg["search"].update(budget=1, population=1)
    path = search(cfg, "Synthetic", "tree_direction", 2021, synthetic=True)
    record = load_records(path)[0]
    assert record["status"] == "complete" and record["proxy"]["status"] == "failed"


def test_proxy_switch_does_not_change_search_decisions(cfg):
    cfg["search"].update(budget=4, population=2)
    a = search(cfg, "Synthetic", "tree_direction", 2021, "proxy_on", synthetic=True)
    disabled = copy.deepcopy(cfg); disabled["proxy"]["enabled"] = False
    b = search(disabled, "Synthetic", "tree_direction", 2021, "proxy_off", synthetic=True)
    ar, br = load_records(a), load_records(b)
    assert [r["hash"] for r in ar] == [r["hash"] for r in br]
    assert [r["objectives"] for r in ar] == [r["objectives"] for r in br]
    assert read_json(a / "representatives.json") == read_json(b / "representatives.json")


@pytest.mark.parametrize("variant", ["chain_forward", "tree_forward", "chain_direction", "tree_direction_random"])
def test_comparison_searches(cfg, variant):
    cfg["search"].update(budget=4, population=2)
    cfg["proxy"]["enabled"] = False
    path = search(cfg, "Synthetic", variant, 2021, synthetic=True)
    assert len(load_records(path)) == 4
    assert all(r["status"] == "complete" for r in load_records(path))


def test_consecutive_failures_stop(cfg, monkeypatch):
    import mamba_tree_nas.search as module
    def fail(*a, **kw):
        raise RuntimeError("backend intentionally unavailable")
    monkeypatch.setattr(module, "train", fail)
    cfg["proxy"]["enabled"] = False
    cfg["search"]["consecutive_failure_limit"] = 2
    with pytest.raises(RuntimeError, match="Consecutive"):
        search(cfg, "Synthetic", "tree_direction", 2021, synthetic=True)
    state = next(Path(cfg["output"]).rglob("search_state.json"))
    assert len(read_json(state)["completed"]) == 2
    assert all(r["status"] == "failed" for r in load_records(state.parent))


def test_candidate_commit_cache_recovers_without_retraining(cfg, monkeypatch):
    # Simulate death after candidate JSON commit but before state advancement.
    import mamba_tree_nas.search as module
    original = module.atomic_json
    fired = [False]
    def interrupt(path, value):
        if Path(path).parent.name == "candidates" and not fired[0]:
            original(path, value); fired[0] = True
            raise KeyboardInterrupt("after candidate commit")
        original(path, value)
    cfg["search"].update(budget=1, population=1)
    monkeypatch.setattr(module, "atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        search(cfg, "Synthetic", "tree_direction", 2021, synthetic=True)
    monkeypatch.setattr(module, "atomic_json", original)
    monkeypatch.setattr(module, "train", lambda *a, **kw: pytest.fail("completed candidate was trained again"))
    path = search(cfg, "Synthetic", "tree_direction", 2021, resume=True, synthetic=True)
    assert len(load_records(path)) == 1


@pytest.mark.skipif(not __import__("torch").cuda.is_available() or not __import__("sys").platform.startswith("linux"), reason="Real Mamba GPU acceptance requires Linux/WSL2 NVIDIA")
def test_real_gpu_forward_backward():
    cfg = load_config()
    assert verify(cfg, True)["gpu_acceptance"] == "passed_real_mamba_forward_backward"


@pytest.mark.skipif(not __import__("torch").cuda.is_available() or not __import__("sys").platform.startswith("linux"), reason="Real functional-tree Mamba acceptance requires Linux/WSL2 NVIDIA")
def test_real_gpu_tree_variable_length():
    import torch
    from mamba_tree_nas.architecture import Architecture, Leaf, Node
    from mamba_tree_nas.model import Classifier
    for direction in ("F", "B", "FB_shared"):
        arch = Architecture("patch8", 64, "meanmax", Node("RES", (Node("OP", leaf=Leaf(direction, 8, 4, 1)),)))
        model = Classifier(arch, 3, 3, 0, "mamba", "cuda").cuda().eval()
        x = torch.randn(2, 17, 3, device="cuda")
        mask = torch.arange(17, device="cuda")[None, :] < torch.tensor([17, 11], device="cuda")[:, None]
        y = model(x, mask)
        padded = torch.nn.functional.pad(x, (0,0,0,15))
        padded_mask = torch.nn.functional.pad(mask, (0,15))
        assert torch.allclose(y, model(padded, padded_mask), atol=2e-5, rtol=2e-5)
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
