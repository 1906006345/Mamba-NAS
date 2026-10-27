import copy
import csv
import random
from pathlib import Path

import pytest

from mamba_tree_nas.architecture import Architecture, Space, is_chain
from mamba_tree_nas.artifacts import atomic_json, read_json
from mamba_tree_nas.cli import main, parser
from mamba_tree_nas.config import digest
from mamba_tree_nas.search import load_records, run_path, search as tree_search
from mamba_tree_nas.traditional import SerialVectorSpace, baseline_config, new_pending, search
from mamba_tree_nas.traditional_compare import comparison_report, matrix


def small_config(cfg):
    cfg["search"].update(population=2, budget=4)
    cfg["train"].update(epochs=1, min_epochs=1)
    cfg["calibration"].update(epochs=1, min_epochs=1)
    cfg["seeds"]["search"] = [2021]
    return cfg


def test_baselines_do_not_replace_or_mutate_tree_experiments(cfg):
    before = digest(cfg)
    c = baseline_config(cfg)
    assert digest(cfg) == before
    assert all(c["experiments"][k] == v for k, v in cfg["experiments"].items())
    assert set(c["experiments"]) - set(cfg["experiments"]) == {"traditional_nsga2", "traditional_random"}
    args = parser().parse_args(["compare-traditional", "--dataset", "Heartbeat", "--search-seed", "2022"])
    assert matrix(c, args) == [("Heartbeat", "traditional_nsga2", 2022), ("Heartbeat", "traditional_random", 2022)]
    args.variant = "tree_direction"
    with pytest.raises(ValueError, match="--variant"):
        matrix(c, args)


def test_conditional_vector_identity_and_operators(cfg, monkeypatch):
    c = baseline_config(cfg)
    space = SerialVectorSpace(c["space"], c["experiments"]["traditional_nsga2"], c["traditional"])
    rng = random.Random(37)
    # The conventional operators must never call tree subtree operators.
    monkeypatch.setattr(Space, "crossover", lambda *a: pytest.fail("tree crossover used by serial baseline"))
    monkeypatch.setattr(Space, "mutate", lambda *a: pytest.fail("tree mutation used by serial baseline"))
    counts, directions = set(), set()
    parents = [space.sample(rng), space.sample(rng)]
    for _ in range(150):
        a = space.mutate(space.crossover(*parents, rng), rng)
        assert is_chain(a.tree) and space.adapter.valid(a)
        vector = space.encode(a)
        assert space.decode(vector).hash == a.hash
        counts.add(a.tree.leaves)
        directions.update(space.choices[i][v] for i, v in enumerate(vector) if i >= 4 and (i-4) % 4 == 0 and v is not None)
        changed = list(vector)
        for i, gene in enumerate(changed):
            if gene is None:
                changed[i] = len(space.choices[i]) - 1
        assert space.decode(changed).hash == a.hash  # inactive slots are not unique architectures
        parents = [parents[1], a]
    assert counts == {1, 2, 3}
    assert directions == {"F", "B", "FB_shared"}


def test_random_sampling_does_not_read_objectives_and_initial_population_matches(cfg, monkeypatch):
    import mamba_tree_nas.traditional as module
    c = baseline_config(cfg)
    spaces = [SerialVectorSpace(c["space"], c["experiments"][v], c["traditional"]) for v in ("traditional_nsga2", "traditional_random")]
    initial = [new_pending(s, c, random.Random(2021), [], [], 3) for s in spaces]
    assert initial[0] == initial[1]
    monkeypatch.setattr(module, "tournament", lambda *a: pytest.fail("Random Search read fitness"))
    monkeypatch.setattr(spaces[1], "crossover", lambda *a: pytest.fail("Random Search crossed parents"))
    monkeypatch.setattr(spaces[1], "mutate", lambda *a: pytest.fail("Random Search mutated parents"))
    pending = new_pending(spaces[1], c, random.Random(9), [], [{"objectives": "must not be inspected"}], 3)
    assert len({Architecture.from_dict(a).hash for a in pending}) == 3


@pytest.mark.parametrize("algorithm", ["nsga2", "random"])
def test_resume_unique_budget_and_source_config_conflict(cfg, algorithm):
    c = baseline_config(small_config(cfg))
    c["proxy"]["enabled"] = False
    variant = c["traditional"]["variants"][algorithm]
    with pytest.raises(InterruptedError):
        search(c, "Synthetic", variant, 2021, "interrupted", synthetic=True, interrupt_after=1)
    p = run_path(c, "Synthetic", variant, 2021, "interrupted")
    first = next((p / "candidates").glob("*.json"))
    snapshot = (first.read_bytes(), first.stat().st_mtime_ns)
    resumed = search(c, "Synthetic", variant, 2021, "interrupted", resume=True, synthetic=True)
    clean = search(c, "Synthetic", variant, 2021, "clean", synthetic=True)
    a, b = load_records(resumed), load_records(clean)
    assert len({r["hash"] for r in a}) == c["search"]["budget"]
    assert [r["hash"] for r in a] == [r["hash"] for r in b]
    assert [r["objectives"] for r in a] == [r["objectives"] for r in b]
    assert (first.read_bytes(), first.stat().st_mtime_ns) == snapshot
    assert all(is_chain(Architecture.from_dict(r["architecture"]).tree) for r in a)
    assert not list((p / "candidates").glob("*.pt"))
    changed = copy.deepcopy(c)
    changed["traditional"]["source_sha256"] = "different-version"
    with pytest.raises(ValueError, match="Resume refused"):
        search(changed, "Synthetic", variant, 2021, "interrupted", resume=True, synthetic=True)


def test_identical_architecture_has_identical_tree_and_baseline_evaluation(cfg, monkeypatch):
    c = baseline_config(small_config(cfg))
    c["search"].update(population=1, budget=1)
    arch = SerialVectorSpace(c["space"], c["experiments"]["traditional_nsga2"], c["traditional"]).sample(random.Random(4))
    monkeypatch.setattr(SerialVectorSpace, "sample", lambda *a: arch)
    monkeypatch.setattr(Space, "sample", lambda *a: arch)
    traditional = search(c, "Synthetic", "traditional_nsga2", 2021, synthetic=True)
    tree = tree_search(c, "Synthetic", "chain_direction", 2021, synthetic=True)
    a, b = load_records(traditional)[0], load_records(tree)[0]
    for key in ("architecture", "validation", "objectives", "parameters", "macs", "best_epoch", "history"):
        assert a[key] == b[key]
    assert a["proxy"]["aggregate"] == b["proxy"]["aggregate"]
    assert read_json(traditional / "representatives.json") == read_json(tree / "representatives.json")
    assert [p["hv"] for p in read_json(traditional / "search_state.json")["history"]] == [p["hv"] for p in read_json(tree / "search_state.json")["history"]]


@pytest.mark.parametrize("algorithm", ["nsga2", "random"])
def test_candidate_commit_before_state_is_recovered(cfg, monkeypatch, algorithm):
    import mamba_tree_nas.traditional as module
    c = baseline_config(small_config(cfg))
    c["search"].update(population=1, budget=1)
    c["proxy"]["enabled"] = False
    variant = c["traditional"]["variants"][algorithm]
    write = module.atomic_json
    def interrupt(path, value):
        write(path, value)
        if Path(path).parent.name == "candidates":
            raise KeyboardInterrupt("after candidate commit")
    monkeypatch.setattr(module, "atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        search(c, "Synthetic", variant, 2021, synthetic=True)
    monkeypatch.setattr(module, "atomic_json", write)
    monkeypatch.setattr(module, "train", lambda *a, **kw: pytest.fail("Committed candidate retrained"))
    p = search(c, "Synthetic", variant, 2021, resume=True, synthetic=True)
    assert len(load_records(p)) == 1


def test_cli_comparison_refit_report_existing_tree_and_skip(cfg, monkeypatch):
    import mamba_tree_nas.cli as cli
    import mamba_tree_nas.report as reporting
    import mamba_tree_nas.traditional as module
    c = small_config(cfg)
    c["search"].update(population=1, budget=2)
    c["logging"]["tensorboard"] = True
    tree = tree_search(c, "Synthetic", "tree_direction", 2021, synthetic=True)
    monkeypatch.setattr(cli, "load_config", lambda *a: copy.deepcopy(c))
    # Still render actual comparison plots, but avoid dozens of per-run figures.
    monkeypatch.setattr(reporting, "_run_figures", lambda *a: None)
    command = ["compare-traditional", "--profile", "smoke", "--dataset", "Synthetic", "--with-refit", "--include-tree"]
    main(command)
    export = Path(c["exports"]) / "traditional_comparison" / "main"
    manifest = read_json(export / "report_manifest.json")
    assert manifest["run_count"] == 3
    assert str(tree / "manifest.json") in manifest["sources"]
    assert (export / "results.csv").exists() and (export / "refit_metrics.csv").exists()
    assert (export / "Synthetic/comparison_hv.pdf").exists()
    assert (export / "Synthetic/comparison_pareto.png").exists()
    with (export / "search_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 3 and all(r["search_count"] == "1" for r in summary)
    assert {r["encoding"] for r in summary} == {"functional_tree", "categorical_serial"}
    snapshots = {}
    for variant in ("traditional_nsga2", "traditional_random"):
        path = run_path(c, "Synthetic", variant, 2021, "main")
        assert read_json(path / "refit/state.json")["status"] == "complete"
        unique = len(read_json(path / "representatives.json")["architectures"])
        assert len(list((path / "refit").rglob("*.pt"))) == unique
        assert len(list((path / "refit").rglob("*_predictions.npz"))) == unique * 3
        assert list((path / "tensorboard/search").glob("events.out.tfevents.*"))
        for file in list((path / "candidates").glob("*.json")) + list((path / "refit").rglob("seed_*.json")):
            snapshots[file] = file.read_bytes(), file.stat().st_mtime_ns
    monkeypatch.setattr(module, "train", lambda *a, **kw: pytest.fail("Completed baseline retrained"))
    main(command + ["--no-report"])
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == before for p, before in snapshots.items())
    main(["compare-traditional", "--profile", "smoke", "--dataset", "Synthetic", "--report-only", "--include-tree"])


def test_dry_run_has_no_data_or_output_side_effects(cfg, monkeypatch, capsys):
    import json
    import mamba_tree_nas.cli as cli
    import mamba_tree_nas.data as data
    monkeypatch.setattr(cli, "load_config", lambda *a: cfg)
    monkeypatch.setattr(data, "DataProtocol", lambda *a: pytest.fail("dry-run loaded data"))
    main(["compare-traditional", "--dry-run", "--with-refit"])
    result = json.loads(capsys.readouterr().out)
    assert result["search_runs"] == 6  # two baselines x three search seeds, synthetic smoke
    assert result["refit_fits_upper_if_refit"] == 54
    assert not Path(cfg["output"]).exists()


@pytest.mark.parametrize("mismatch", ["hv", "directions"])
def test_comparison_rejects_unequal_conventions_before_export(cfg, mismatch):
    c = baseline_config(small_config(cfg))
    c["search"].update(population=1, budget=1)
    c["proxy"]["enabled"] = False
    runs = [("Synthetic", v, 2021) for v in ("traditional_nsga2", "traditional_random")]
    paths = [search(c, d, v, s, synthetic=True) for d, v, s in runs]
    if mismatch == "hv":
        state = read_json(paths[1] / "search_state.json")
        state["cost_bound"] *= 2
        atomic_json(paths[1] / "search_state.json", state)
    else:
        manifest = read_json(paths[1] / "manifest.json")
        manifest["config"]["experiments"]["traditional_random"]["directions"] = ["F"]
        atomic_json(paths[1] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="scan directions or HV normalization"):
        comparison_report(c, runs, "main", cfg["exports"])
    assert not Path(cfg["exports"]).exists()


def test_search_summary_uses_independent_complete_searches_only(cfg, monkeypatch):
    import mamba_tree_nas.traditional_compare as module
    c = baseline_config(cfg)
    c["search"].update(population=1, budget=1)
    runs = [("Synthetic", "traditional_nsga2", seed) for seed in (2021, 2022, 2023)]
    for (dataset, variant, seed), hv in zip(runs, (.2, .6, None)):
        path = run_path(c, dataset, variant, seed, "main")
        identity = {"dataset": dataset, "variant": variant, "search_seed": seed, "run_id": "main", "synthetic": True}
        complete = hv is not None
        atomic_json(path / "manifest.json", {"config": c, "identity": identity})
        history = [{"candidate_count": 1, "elapsed_seconds": 10, "hv": hv, "pareto_size": 1, "failure_rate": 0, "topology_count": 1, "topology_entropy": 0}] if complete else []
        atomic_json(path / "search_state.json", {"status": "complete" if complete else "interrupted", "completed": ["a"] if complete else [], "cost_bound": 100, "history": history, "elapsed_seconds": 10})
        if complete:
            atomic_json(path / "candidates/a.json", {"status": "complete", "proxy": {"seconds": 1}, "validation": {"macro_f1": hv}})
    monkeypatch.setattr(module, "report", lambda *a, **kw: None)
    export = comparison_report(c, runs, "main", cfg["exports"])
    with (export / "search_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["search_count"] == "2" and row["selected_search_count"] == "3"
    assert row["all_selected_searches_complete"] == "False"
    assert float(row["final_hv_mean"]) == pytest.approx(.4)
    assert float(row["final_hv_std"]) == pytest.approx(2 ** .5 * .2)
