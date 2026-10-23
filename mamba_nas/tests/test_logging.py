from __future__ import annotations

import logging

from mamba_nas.logging_utils import configure_logging, log_test_result


def test_process_logs_and_test_results_are_separated(tmp_path):
    process_path = tmp_path / "search.log"
    result_path = tmp_path / "result.txt"
    configure_logging(process_path)
    logging.getLogger("mamba_nas.search").info("[搜索][测试] candidate=abc")
    log_test_result(
        "Heartbeat",
        "knee",
        2021,
        {
            "candidate_hash": "abcdef1234567890",
            "macro_f1": 0.8,
            "accuracy": 0.75,
            "parameters": 100,
            "macs": 200,
            "training_seconds": 1.5,
        },
        aliased=False,
        result_path=result_path,
    )
    for handler in logging.getLogger("mamba_nas").handlers:
        handler.flush()

    process_text = process_path.read_text(encoding="utf-8")
    result_text = result_path.read_text(encoding="utf-8")
    assert "[搜索][测试] candidate=abc" in process_text
    assert "[TEST最终结果]" not in process_text
    assert "[TEST最终结果] dataset=Heartbeat role=knee seed=2021" in result_text
    assert "[搜索][测试]" not in result_text
    assert "macro_f1=0.800000 accuracy=0.750000" in result_text
