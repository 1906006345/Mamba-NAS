import json

from mamba_nas.search import CandidateEvaluator, nondominated, select_representatives


def test_pareto_and_representative_selection():
    rows = [
        {"candidate_hash": "a", "macro_f1": 0.9, "macs": 100, "objective_f1": 0.1, "objective_macs": 4.6},
        {"candidate_hash": "b", "macro_f1": 0.8, "macs": 50, "objective_f1": 0.2, "objective_macs": 3.9},
        {"candidate_hash": "c", "macro_f1": 0.7, "macs": 150, "objective_f1": 0.3, "objective_macs": 5.0},
    ]
    front = nondominated(rows)
    assert {row["candidate_hash"] for row in front} == {"a", "b"}
    selected = select_representatives(front)
    assert selected["high_accuracy"]["candidate_hash"] == "a"
    assert selected["low_cost"]["candidate_hash"] == "b"


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
