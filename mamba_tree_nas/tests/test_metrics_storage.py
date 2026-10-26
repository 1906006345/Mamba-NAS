import copy
import math
import random
import numpy as np
import pytest
from mamba_tree_nas.architecture import Architecture, Leaf, Node, Space
from mamba_tree_nas.artifacts import atomic_json, read_json, run_lock, write_csv
from mamba_tree_nas.cost import cost_upper_bound, leaf_macs, macs
from mamba_tree_nas.metrics import classification, crowding, fronts, hypervolume, nested_summary, nsga_select, proxy_diagnostics, representatives
from mamba_tree_nas.report import external_baselines, hv_auc


def test_classification_fixed_class_set():
    result = classification([0, 0, 1, 1], [[.8,.2,0], [.3,.7,0], [.2,.8,0], [.1,.9,0]], 3)
    assert result["accuracy"] == .75
    assert result["macro_f1"] == pytest.approx((2/3 + .8) / 3)
    assert result["balanced_accuracy"] == .5
    assert result["per_class"][2]["f1"] == 0
    assert result["confusion_matrix"] == [[1,1,0], [0,2,0], [0,0,0]]


def test_fronts_crowding_hv_and_representatives():
    points = [[.2,.8], [.5,.4], [.8,.2], [.9,.9], [.2,.8]]
    assert set(fronts(points)[0]) == {0,1,2,4}
    assert hypervolume(points, (1,1)) == pytest.approx(.2*.8 + .3*.6 + .3*.2)
    assert hypervolume([[2, 2]], (1,1)) == 0
    records = [{"hash": str(i), "status": "complete", "objectives": p, "macs": int(math.expm1(p[1]) * 100)} for i, p in enumerate(points)]
    assert len(nsga_select(records, 3)) == 3
    selected = representatives(records, 100)
    assert selected["high_f1"] == "0" and selected["low_macs"] == "2"
    for point in ([.3,.6], [.1,.9]):
        assert hypervolume(points + [point]) >= hypervolume(points)


def test_hv_auc_known_step_example():
    rows = [{"candidate_count":1, "hv":.2}, {"candidate_count":2, "hv":.5}, {"candidate_count":3, "hv":.6}]
    assert hv_auc(rows, "candidate_count")["auc"] == pytest.approx(.7)


def test_recursive_macs_sharing_and_bound(cfg):
    f = Node("OP", leaf=Leaf("F", 2, 2, 1))
    fb = Node("OP", leaf=Leaf("FB_shared", 2, 2, 1))
    a = Architecture("point", 8, "mean", f)
    parallel = Architecture("point", 8, "mean", Node("PAR", (f, f)))
    sequential = Architecture("point", 8, "mean", Node("SEQ", (f, f)))
    shared = Architecture("point", 8, "mean", fb)
    assert macs(parallel, 17, 3, 3) == macs(sequential, 17, 3, 3)
    assert macs(shared, 17, 3, 3) - macs(a, 17, 3, 3) == leaf_macs(8, 17, f.leaf)
    upper = cost_upper_bound(cfg["space"], 17, 3, 3)
    rng = random.Random(9)
    for variant in cfg["experiments"].values():
        space = Space(cfg["space"], variant)
        assert all(macs(space.sample(rng), 17, 3, 3) <= upper for _ in range(50))


def test_nested_statistics():
    rows = []
    for seed, scores in [(2021, [.1,.2,.3]), (2022, [.5,.6,.7]), (2023, [.7,.8,.9])]:
        for fit, score in enumerate(scores):
            rows.append({"dataset":"A", "variant":"tree", "role":"knee", "search_seed":seed, "refit_seed":fit, **{k:score for k in ("macro_f1","accuracy","balanced_accuracy","ce")}})
    within, outer = nested_summary(rows)
    assert len(within) == 3 and len(outer) == 1
    assert outer[0]["search_count"] == 3
    assert outer[0]["macro_f1_mean"] == pytest.approx(np.mean([.2,.6,.8]))
    assert outer[0]["macro_f1_between_std"] == pytest.approx(np.std([.2,.6,.8], ddof=1))
    assert outer[0]["macro_f1_mean_within_std"] == pytest.approx(.1)


def test_external_provenance(tmp_path):
    path = tmp_path / "external.csv"
    base = {"dataset":"A", "method":"Published", "source":"https://example.org/paper", "provenance":"literature", "protocol_description":"Original paper protocol", "accuracy":.8}
    write_csv(path, [base])
    assert "not established" in external_baselines(path, {})[0]["comparability"]
    write_csv(path, [dict(base, provenance="same_protocol")])
    with pytest.raises(ValueError, match="fingerprint"):
        external_baselines(path, {})
    write_csv(path, [dict(base, provenance="same_protocol", evaluation_fingerprint="abc")])
    assert external_baselines(path, {"A":{"abc"}})[0]["provenance"] == "same_protocol"


def test_atomic_json_and_run_lock(tmp_path):
    path = tmp_path / "a.json"
    atomic_json(path, {"a":1}); atomic_json(path, {"a":2})
    assert read_json(path) == {"a":2}
    with pytest.raises(ValueError):
        atomic_json(path, {"a":float("nan")})
    assert read_json(path) == {"a":2}
    assert not list(tmp_path.glob("*.tmp"))
    with run_lock(tmp_path / ".lock"):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path / ".lock"):
                pass


def test_proxy_rank_both_directions():
    records = [{"status":"complete", "validation":{"macro_f1":i/10}, "proxy":{"status":"complete", "aggregate":{"full":{"product":i+1}}}} for i in range(10)]
    r = next(x for x in proxy_diagnostics(records) if x["scope"] == "full" and x["field"] == "product")
    assert r["spearman"] == pytest.approx(1)
    assert r["ascending_hit"] == 0 and r["descending_hit"] == 1


def test_invalid_proxy_scope_not_used_for_rank():
    record = {"status":"complete", "validation":{"macro_f1":.9}, "proxy":{"status":"complete", "aggregate":{"full":{"gradient_l2":999}}, "repeats":[{"full":{"status":"missing_gradient"}}]}}
    rows = proxy_diagnostics([record])
    assert next(r for r in rows if r["scope"] == "full" and r["field"] == "gradient_l2")["n"] == 0
