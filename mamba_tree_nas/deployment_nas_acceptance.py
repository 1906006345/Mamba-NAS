"""Check that the deployment suite produced usable outputs in all four stages."""
import argparse
from pathlib import Path

from .artifacts import read_json
from .deployment_report import collect, pair_metrics
from .deployment_search import ARMS, SUPERNET


def verify(output, synthetic=False):
    protocol = read_json(output / "protocol.json")
    mode = "synthetic_cpu_smoke" if synthetic else "cuda"
    assert protocol["runtime"]["mode"] == mode
    settings = protocol["config"]["deployment_nas"]
    seed = settings["search_seeds"][0]
    root = output / "Heartbeat"
    assert read_json(root / "data.json")["test_accessed"] is False
    panel = read_json(root / "panel.json")
    assert len(panel["architectures"]) == settings["panel_size"]
    assert read_json(root / "benchmark" / "summary.json")["valid"] == settings["panel_size"]
    assert read_json(root / "rank" / "summary.json")["valid_pairs"] == settings["panel_size"]
    bank = read_json(root / "supernet" / str(seed) / "summary.json")
    assert bank["epochs"] == settings["supernet_epochs"]
    assert bank["export_max_abs_difference"] < 1e-5
    assert any(bank["activation_counts"].values())
    report, online, audited = collect(output)
    for arm in (*ARMS, SUPERNET):
        key = ("Heartbeat", seed, arm)
        state = read_json(root / "search" / arm / str(seed) / "state.json")
        assert state["status"] == "complete" and len(online[key]) == settings["search"]["budget"]
        assert all(r["status"] == "complete" for r in state["records"])
        for row in state["records"]:
            hw = row["hardware"]
            assert hw["latency_ms"] > 0 and hw["peak_bytes"] > 0
            if not synthetic:
                assert hw["cuda_event_p50_ms"] > 0
                assert hw["request_p50_ms"] > 0 and hw["throughput_samples_per_second"] > 0
                assert hw["memory_kind"] == "cuda_peak_allocated_minus_pre_model_baseline"
        if arm in (ARMS[2], SUPERNET):
            assert [r["candidates"] for r in audited[key]] == settings["checkpoints"]
            assert audited[key][-1]["failed_audits"] == 0
        if arm == SUPERNET:
            assert online[key][0]["wall_seconds"] >= bank["elapsed_seconds"] > 0
    pairs = pair_metrics(report, online, audited)
    assert {row["stage"] for row in pairs} == {"objectives", "compare"}
    assert (output / "report" / "Heartbeat" / f"audited_hv_{seed}.png").is_file()
    assert (output / "report" / "Heartbeat" / f"f1_latency_{seed}.png").is_file()
    assert (output / "report" / "Heartbeat" / "rank.png").is_file()
    print(f"Verified deployment objectives, tree ranking and comparison ({mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    verify(args.output, synthetic=args.synthetic)
