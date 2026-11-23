import sys

import pytest
import torch

from mamba_tree_nas.architecture import Architecture, Leaf, Node
from mamba_tree_nas.artifacts import read_json
from mamba_tree_nas.data import DataProtocol, collate
from mamba_tree_nas.deployment_benchmark import benchmark
from mamba_tree_nas.deployment_metrics import curves_at, hypervolume, nsga_order, rank_summary
from mamba_tree_nas.deployment_nas import main
from mamba_tree_nas.deployment_nas_acceptance import verify
from mamba_tree_nas.deployment_protocol import Study, bind_study, configuration, runtime_identity
from mamba_tree_nas.deployment_search import ARMS, run_search
from mamba_tree_nas.metrics import hypervolume as previous_hypervolume


def test_3d_hypervolume_and_2d_compatibility():
    points = [(0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    assert hypervolume(points + [points[0], (.6, .6, .6), (0, 0, 2)], (1, 1, 1)) == pytest.approx(.5)
    two_d = [(0.2, .8), (.7, .1)]
    assert hypervolume(two_d) == previous_hypervolume(two_d)
    assert hypervolume([(x, y, .25) for x, y in two_d], (1.1, 1.1, 1)) == pytest.approx(
        previous_hypervolume(two_d) * .75)
    with pytest.raises(ValueError, match="dimension"):
        hypervolume(two_d, (1, 1, 1))


def test_crowding_uses_third_axis_and_curves_share_time_limit():
    rows = [{"hash": key, "objectives": point} for key, point in (
        ("a", (0, 3, 1)), ("b", (1, 2, 0)), ("c", (2, 1, 2)), ("d", (3, 0, 1)))]
    assert [r["hash"] for r in nsga_order(rows)] == ["a", "b", "c", "d"]
    history = [{"wall": 2, "hv": .5}, {"wall": 6, "hv": .8}]
    assert curves_at(history, "wall", 4) == pytest.approx(.25)
    assert curves_at(history, "wall", 8) == pytest.approx(.45)


def test_rank_ties_failures_and_f1_regret():
    rows = [{"hash": "a", "short_f1": .9, "reference_f1": .8},
            {"hash": "b", "short_f1": .9, "reference_f1": .9},
            {"hash": "c", "short_f1": .5, "reference_f1": .7},
            {"hash": "d", "short_f1": .1, "reference_f1": None}]
    result = rank_summary(rows)
    assert result["valid_pairs"] == 3 and result["failed_pairs"] == 1
    assert result["short_f1_tied_architectures"] == 2
    assert 0 < result["kendall_tau_b"] < 1
    assert result["spearman"] == pytest.approx(3 ** .5 / 2)
    assert result["selection_regret_f1_pp"] == pytest.approx(10)
    assert rank_summary(rows[:2])["kendall_tau_b"] is None


def test_search_interruption_resumes_with_existing_scores(tmp_path):
    output = tmp_path / "r"
    output.mkdir()
    cfg = configuration("smoke")
    signature = bind_study(output, cfg, runtime_identity(cfg))
    study = Study(cfg, output, "Heartbeat", signature)
    benchmark(study)
    seed = 52021
    with pytest.raises(KeyboardInterrupt):
        run_search(study, ARMS[0], seed, interrupt_after=2)
    path = study.root / "search" / ARMS[0] / str(seed) / "state.json"
    interrupted = read_json(path)
    assert interrupted["status"] == "interrupted" and len(interrupted["pending"]) == 1
    resumed = run_search(study, ARMS[0], seed)
    assert resumed["status"] == "complete"
    assert resumed["records"][:2] == interrupted["records"]
    other = run_search(study, ARMS[1], seed)
    assert [r["hash"] for r in resumed["records"][:3]] == [r["hash"] for r in other["records"][:3]]
    assert all(r["score_cache_hit"] for r in other["records"][:3])
    cfg["deployment_nas"]["benchmark"]["warmup"] += 1
    with pytest.raises(ValueError, match="protocol"):
        bind_study(output, cfg, runtime_identity(cfg))


def test_supernet_subnet_semantics_and_inactive_parameters(tmp_path):
    from mamba_tree_nas.deployment_supernet import TreeSupernet

    cfg = configuration("smoke")
    study = Study(cfg, tmp_path, "PEMS-SF", "unit-test")
    bank = TreeSupernet(study)
    leaf = Node("OP", leaf=Leaf("FB_shared", 2, 2, 1, 2))
    tree = Node("PAR", (Node("RES", (leaf,)), Node("SEQ", (leaf, leaf))))
    arch = Architecture("point", 8, "meanmax", tree, 2)
    path = bank.path(arch)
    assert len([k for k in path.activated if k.startswith("mixers/")]) == 3
    assert path.tree.branches[0] is not path.tree.branches[1]
    x, mask, _, _ = collate([study.data.train[0], study.data.train[1]])
    path(x, mask).mean().backward()
    assert bank.tokenizers["8_point"].projection.weight.grad is not None
    assert bank.tokenizers["8_patch8"].projection.weight.grad is None
    bank.eval()
    exported = bank.export(arch)
    with torch.inference_mode():
        torch.testing.assert_close(bank.path(arch)(x, mask), exported(x, mask))
    original = bank.tokenizers["8_point"].projection.weight.detach().clone()
    with torch.no_grad():
        exported.tokenizer.projection.weight.zero_()
    torch.testing.assert_close(bank.tokenizers["8_point"].projection.weight, original)


def test_cli_full_cpu_suite_without_test_or_supernet_for_rank(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("TEST was accessed")

    monkeypatch.setattr(DataProtocol, "load_test", forbidden)
    output = tmp_path / "s"
    common = ["--profile", "smoke", "--output", str(output)]
    main(["compare", *common, "--dry-run"])
    assert not output.exists()
    for stage in ("benchmark", "objectives"):
        main([stage, *common, "--execute"])
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "mamba_tree_nas.deployment_supernet", None)
        main(["rank", *common, "--execute"])
    assert not (output / "Heartbeat" / "supernet").exists()
    for stage in ("compare", "report"):
        main([stage, *common, "--execute"])
    verify(output, synthetic=True)
    main(["compare", *common, "--execute"])
    verify(output, synthetic=True)
