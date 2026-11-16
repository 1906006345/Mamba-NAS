from dataclasses import replace
import csv
import random

import pytest

from mamba_tree_nas.architecture import Architecture, Leaf, Node, Space, at
from mamba_tree_nas.artifacts import read_json
from mamba_tree_nas.config import load_config
from mamba_tree_nas.cost import trunk_macs
from mamba_tree_nas.search import _new_pending, load_records, run_path, search
from mamba_tree_nas.temporal_ops import neighbors
from mamba_tree_nas.temporal_study import main, under_cap


def op(rate):
    return Node("OP", leaf=Leaf("F", 2, 2, 1, rate))


def setup():
    cfg = load_config(profile="smoke", multiscale=True, temporal=True)
    variant = "tree_multiscale_forward"
    return cfg, Space(cfg["space"], cfg["experiments"][variant], 405)


def test_cost_windows_and_canonical_deduplication():
    _, space = setup()
    dims = (405, 61, 2)
    a = Architecture("patch8", 8, "mean", Node("PAR", (op(2), Node("RES", (op(4),)))), 2)
    children, _ = neighbors(space, a, "redistribute", dims, 0.15)
    swapped, = children.values()
    assert at(swapped.tree, (0,)).leaf.rate == 4 and at(swapped.tree, (1, 0)).leaf.rate == 2
    assert swapped.tree.shape == a.tree.shape and trunk_macs(swapped, 405) == trunk_macs(a, 405)
    assert neighbors(space, a, "redistribute", dims, 0.15, children)[1]["already_seen"] == 1
    symmetric = replace(a, tree=Node("PAR", (op(2), op(4))))
    assert neighbors(space, symmetric, "redistribute", dims, 0.15)[1]["equivalent"] == 1
    single = replace(a, tree=op(1))
    assert neighbors(space, single, "single", dims, 1)[0].popitem()[1].tree.leaf.rate == 2
    assert neighbors(space, single, "single", dims, 0)[1]["outside_window"] == 1
    short = Space(space.c, space.e, 25)
    assert not neighbors(short, single, "single", (25, 61, 2), 1)[0]


def test_disabled_rng_and_fallback():
    cfg, space = setup()
    cfg["search"]["mutation_weights"] = {"leaf": 1}
    cfg["search"].update(crossover_probability=0, mutation_probability=1)
    a = Architecture("patch8", 8, "mean", op(1), 2)
    row = {"hash": a.hash, "architecture": a.to_dict(), "objectives": [0.5, 1.0]}
    old = {**cfg, "search": {k: v for k, v in cfg["search"].items() if k != "temporal_mutation"}}
    r1, r2 = random.Random(12), random.Random(12)
    assert _new_pending(space, cfg, r1, [row], [row], 1) == _new_pending(space, old, r2, [row], [row], 1)
    assert r1.getstate() == r2.getstate()
    cfg["search"]["temporal_mutation"].update(enabled=True, single_probability=0, redistribute_probability=1)
    telemetry = {"operators": {}, "edits": {}}
    assert _new_pending(space, cfg, r1, [row], [row], 1, dimensions=(405, 61, 2), telemetry=telemetry)
    assert telemetry["operators"]["redistribute"]["fallback_no_neighbor"] >= 1
    assert not telemetry["edits"]


def test_diagnosis_never_reads_test_and_reuses_cache(tmp_path, monkeypatch):
    from mamba_tree_nas.data import DataProtocol
    from mamba_tree_nas import training

    def forbidden(*_):
        pytest.fail("TEST must not be loaded for diagnostics")
    monkeypatch.setattr(DataProtocol, "load_test", forbidden)
    output = tmp_path / "d"
    argv = ["diagnose", "--profile", "smoke", "--output", str(output)]
    main(argv)
    assert not output.exists()
    main(argv + ["--execute"])
    rows = list(csv.DictReader((output / "diagnostic_children.csv").open(encoding="utf-8-sig")))
    assert rows and {r["protocol"] for r in rows} == {"fixed_end", "search_best"}
    monkeypatch.setattr(training, "train", forbidden)
    main(argv + ["--execute"])


def test_search_resume_preserves_batch_and_telemetry(tmp_path):
    cfg, _ = setup()
    cfg["output"] = str(tmp_path / "s")
    cfg["logging"]["tensorboard"] = False
    cfg["search"].update(population=2, budget=6, mutation_weights={"leaf": 1}, mutation_probability=1)
    cfg["search"]["temporal_mutation"].update(enabled=True, window=1, single_probability=1, redistribute_probability=0)
    variant = "tree_multiscale_forward"
    with pytest.raises(InterruptedError):
        search(cfg, "S", variant, 2021, "a", synthetic=True, interrupt_after=3)
    a = run_path(cfg, "S", variant, 2021, "a")
    assert read_json(a / "search_state.json")["pending"]
    search(cfg, "S", variant, 2021, "a", resume=True, synthetic=True)
    b = search(cfg, "S", variant, 2021, "b", synthetic=True)
    a_state, b_state = [read_json(p / "search_state.json") for p in (a, b)]
    assert a_state["completed"] == b_state["completed"]
    assert a_state["temporal_mutation"]["operators"] and a_state["temporal_mutation"] == b_state["temporal_mutation"]
    assert under_cap(load_records(a), 1) == (0, None)


def test_pair_exports_shared_initial_population_and_miss(tmp_path):
    output = tmp_path / "p"
    main(["search", "--profile", "smoke", "--dataset", "Heartbeat", "--seed", "2021", "--output", str(output),
          "--set", "temporal_study.datasets.Heartbeat.cap=1", "--with-refit", "--execute"])
    pair = read_json(output / "pairs" / "Heartbeat" / "2021.json")
    old, new = pair["results"][0]
    assert old["initial_hashes"] == new["initial_hashes"]
    assert not old["hit"] and not new["hit"] and not (output / "refits").exists()
    assert (output / "search_paired.csv").exists()
