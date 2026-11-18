from copy import deepcopy
import csv
from pathlib import Path
import random

import pytest

from mamba_tree_nas.architecture import Architecture, Leaf, Node
from mamba_tree_nas.artifacts import atomic_json, read_json
from mamba_tree_nas.data import DataProtocol
from mamba_tree_nas.followup_evaluation import EvaluationCache
from mamba_tree_nas.followup_search import pending_batch, report, run_search
from mamba_tree_nas.temporal_followup import configuration, data_for, diagnostic_plan, main, source_pool, space_for


def config():
    cfg = configuration(profile="smoke")
    cfg["logging"]["tensorboard"] = False
    return cfg


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected training/TEST/source mutation")


def test_dry_run_matrix_and_config(capsys, tmp_path):
    out = tmp_path / "out"
    main(["search", "--output", str(out), "--with-refit", "--dry-run"])
    printed = capsys.readouterr().out
    assert '"search_runs": 45' in printed
    assert '"candidate_evaluations_upper": 4320' in printed
    assert '"refit_fits_upper": 135' in printed
    assert not out.exists()
    cfg = configuration(profile="smoke", overrides=["temporal_followup.search.budget=7"])
    assert cfg["search"]["budget"] == 7
    cfg = configuration(profile="smoke", overrides=["search.budget=8"])
    assert cfg["search"]["budget"] == 8
    with pytest.raises(ValueError, match="window"):
        configuration(overrides=["temporal_followup.window=-1"])
    with pytest.raises(SystemExit):
        main(["search", "--source-root", str(out), "--output", str(out / "nested")])


def test_cache_identity_excludes_policy_and_preserves_training_protocol(tmp_path, monkeypatch):
    cfg = config()
    data, space, pool, _ = source_pool(cfg, "Heartbeat", tmp_path)
    a = Architecture.from_dict(pool[0]["architecture"])
    cache = EvaluationCache(tmp_path / "cache", cfg, data)
    result, meta = cache.measure(a, 11001, "arm_a")
    assert not meta["hit"] and result["status"] == "complete"
    changed = deepcopy(cfg)
    changed["temporal_followup"]["window"] = 0.3
    changed["seeds"]["search"] = [999]
    other = EvaluationCache(tmp_path / "cache", changed, data)
    assert cache.identity(a, 11001) == other.identity(a, 11001)
    import mamba_tree_nas.training as training
    monkeypatch.setattr(training, "train", forbidden)
    reused, m = other.measure(a, 11001, "arm_b")
    assert m["hit"] and m["first_owner"] == "arm_a" and reused == result
    changed["train"]["learning_rate"] *= 2
    assert cache.identity(a, 11001) != EvaluationCache(tmp_path, changed, data).identity(a, 11001)
    assert cache.identity(a, 11001) != cache.identity(a, 11002)


def test_census_and_diagnostics_never_load_test(tmp_path, monkeypatch):
    monkeypatch.setattr(DataProtocol, "load_test", forbidden)
    output = tmp_path / "out"
    main(["census", "--profile", "smoke", "--dataset", "Heartbeat", "--output", str(output), "--execute"])
    assert (output / "census" / "summary.csv").exists()
    main(["diagnose", "--profile", "smoke", "--dataset", "Heartbeat", "--output", str(output), "--execute"])
    plan = read_json(output / "diagnose" / "plan.json")["Heartbeat"]
    assert plan["parents"] and len(plan["parents"]) <= 2
    for entry in plan["parents"]:
        assert entry["quota"] > 0
        assert entry["children"]["single"] != entry["children"]["matched_random"]
    rows = list(csv.DictReader((output / "diagnose" / "parents.csv").open(encoding="utf-8-sig")))
    assert {r["protocol"] for r in rows} == {"search_best", "fixed_end"}
    import mamba_tree_nas.followup_evaluation as ev
    monkeypatch.setattr(ev, "fixed_train", forbidden)
    main(["diagnose", "--profile", "smoke", "--dataset", "Heartbeat", "--output", str(output), "--execute"])
    assert not list(output.rglob("*.pt"))


def test_gate_fallback_consumes_no_extra_rng(tmp_path, monkeypatch):
    cfg = config()
    cfg["search"].update(mutation_probability=1, mutation_weights={"leaf": 1})
    data, space, pool, _ = source_pool(cfg, "Heartbeat", tmp_path)
    a = Architecture.from_dict(pool[0]["architecture"])
    record = dict(hash=a.hash, architecture=a.to_dict(), objectives=[0.5, 10], status="complete")
    import mamba_tree_nas.temporal_ops as ops
    monkeypatch.setattr(ops, "neighbors", lambda *a, **kw: ({}, {}))
    dims = data.max_length, data.channels, len(data.classes)
    results = []
    for arm in ("original", "matched_random", "single"):
        rng, oprng = random.Random(41), random.Random(91)
        previous = oprng.getstate()
        counters = dict(attempts=0, eligible=0, fallback_no_neighbor=0, submitted=0)
        result = pending_batch(space, cfg, arm, rng, oprng, [record], [record], 2, dims, counters)
        results.append((result, rng.getstate()))
        assert oprng.getstate() == previous
    assert results[0] == results[1] == results[2]


def test_local_edit_is_submitted_without_secondary_probability(tmp_path):
    cfg = config()
    cfg["search"].update(mutation_probability=1, crossover_probability=0, mutation_weights={"leaf": 1})
    data = data_for(cfg, "Heartbeat")
    space = space_for(cfg, "Heartbeat", data)
    a = Architecture("patch8", 8, "mean", Node("OP", leaf=Leaf("F", 2, 2, 1, 1)), 2)
    record = dict(hash=a.hash, architecture=a.to_dict(), objectives=[0.5, 10], status="complete")
    telemetry = dict(attempts=0, eligible=0, fallback_no_neighbor=0, submitted=0)
    pending = pending_batch(space, cfg, "single", random.Random(1), random.Random(2), [record], [record], 1,
                            (data.max_length, data.channels, len(data.classes)), telemetry)
    assert telemetry["attempts"] == telemetry["eligible"] == telemetry["submitted"] == 1
    child = Architecture.from_dict(pending[0]["architecture"])
    assert child.tree.leaf.rate == 2 and pending[0]["edit"]["operator"] == "single"


def test_saved_source_is_read_only_and_mismatch_rejected(tmp_path, monkeypatch):
    import mamba_tree_nas.temporal_followup as followup
    cfg = config()
    data, _, fixture, _ = source_pool(cfg, "Heartbeat", tmp_path)
    cfg["profile"] = "paper"  # Use the real saved-source branch with a TRAIN fixture.
    monkeypatch.setattr(followup, "data_for", lambda *a: data)
    source = tmp_path / "source"
    folder = source / "Heartbeat" / "tm_original" / "seed_2021" / "main"
    a = Architecture.from_dict(fixture[0]["architecture"])
    manifest = dict(identity=dict(dataset="Heartbeat", variant="tree_multiscale_forward", search_seed=2021),
                    config=cfg, data=data.manifest)
    atomic_json(folder / "manifest.json", manifest)
    atomic_json(folder / "search_state.json", dict(status="complete", completed=[a.hash]))
    from mamba_tree_nas.cost import macs
    atomic_json(folder / "candidates" / f"{a.hash}.json", dict(status="complete", architecture=a.to_dict(),
                macs=macs(a, data.max_length, data.channels, len(data.classes)), validation={"macro_f1": .5}))
    before = {str(p): p.read_bytes() for p in source.rglob("*.json")}
    _, _, loaded, _ = source_pool(cfg, "Heartbeat", source)
    assert len(loaded) == 1 and loaded[0]["hash"] == a.hash
    assert before == {str(p): p.read_bytes() for p in source.rglob("*.json")}
    manifest["data"] = dict(data.manifest, split_seed=-1)
    atomic_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="TRAIN/split mismatch"):
        source_pool(cfg, "Heartbeat", source)


def test_resume_common_scores_and_shared_refit(tmp_path, monkeypatch):
    cfg = config()
    cfg["space"].update(d_state=[2, 4], expand=[1, 2])
    cfg["search"].update(population=2, budget=5, mutation_probability=1, mutation_weights={"leaf": 1})
    monkeypatch.setattr(DataProtocol, "load_test", forbidden)
    output = tmp_path / "runs"
    with pytest.raises(InterruptedError):
        run_search(cfg, "Heartbeat", "single", 52021, output, interrupt_after=3)
    path = run_search(cfg, "Heartbeat", "single", 52021, output)
    clean = run_search(cfg, "Heartbeat", "single", 52021, tmp_path / "clean")
    ss, cs = [read_json(p / "search_state.json") for p in (path, clean)]
    assert ss["completed"] == cs["completed"] and ss["mutation"] == cs["mutation"]
    before = (path / "search_state.json").read_bytes()
    run_search(cfg, "Heartbeat", "single", 52021, output)
    assert before == (path / "search_state.json").read_bytes()
    original = run_search(cfg, "Heartbeat", "original", 52021, output)
    def initial(p):
        records = [read_json(x) for x in (p / "candidates").glob("*.json")]
        return sorted((r["hash"], r["validation"]) for r in records if r["generation"] == 0)
    assert initial(path) == initial(original)
    assert all(read_json(p)["score_cache"]["hit"] for p in (original / "candidates").glob("*.json") if read_json(p)["generation"] == 0)
    snapshot = report(output, write=True)
    assert snapshot["complete"] == 2
    assert snapshot["synthetic_warning"]
    changed = deepcopy(cfg)
    changed["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="Resume refused"):
        run_search(changed, "Heartbeat", "single", 52021, output)


def test_refit_alias_reuse_and_nested_summary(tmp_path, monkeypatch):
    import mamba_tree_nas.followup_search as search_module
    cfg = config()
    cfg["search"].update(population=1, budget=1)
    cfg["temporal_study"]["datasets"]["Heartbeat"]["cap"] = 10**12
    out = tmp_path / "runs"
    search_module.run_suite(cfg, ["Heartbeat"], [52021], out, True, True)
    unique = list((out / "shared_refits").glob("Heartbeat/*/refit/*/seed_*.json"))
    assert len(unique) == 3
    assert len(list((out / "shared_refits").rglob("*.pt"))) == 1
    assert len(list((out / "search").rglob("refit_alias.json"))) == 3
    import mamba_tree_nas.training as training
    import mamba_tree_nas.refit as refit_module
    monkeypatch.setattr(training, "train", forbidden)
    monkeypatch.setattr(refit_module, "train", forbidden)
    search_module.run_suite(cfg, ["Heartbeat"], [52021], out, True, True)
    result = search_module.report(out, write=True)
    assert result["unique_refit_fits"] == 3
    assert all(r["test_f1_n"] == 1 and r["test_f1_sd_between_searches"] is None for r in result["summary"])


def test_independent_process_audit_reuse(tmp_path):
    output = tmp_path / "audit"
    args = ["audit", "--profile", "smoke", "--dataset", "Heartbeat", "--output", str(output),
            "--set", "logging.tensorboard=false", "--set", "temporal_followup.audit.architectures=1", "--execute"]
    main(args)
    jobs = read_json(output / "audit" / "plan.json")
    results = [read_json(j["result_path"]) for j in jobs]
    assert len(results) == 2 and results[0]["pid"] != results[1]["pid"]
    assert results[0]["initial_weights_hash"] == results[1]["initial_weights_hash"]
    assert results[0]["final_weights_hash"] == results[1]["final_weights_hash"]
    assert all(not r["test_accessed"] for r in results)
    before = [Path(j["result_path"]).read_bytes() for j in jobs]
    main(args)
    assert before == [Path(j["result_path"]).read_bytes() for j in jobs]
    changed = args + ["--set", "temporal_followup.audit.epochs=3"]
    with pytest.raises(ValueError, match="Protocol changed"):
        main(changed)
