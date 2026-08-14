import json
from pathlib import Path

from mamba_nas.export import export_paper


def test_export_is_regenerated_without_weights(tmp_path: Path):
    run = tmp_path / "runs" / "example"
    dataset = run / "Heartbeat"
    (dataset / "search" / "generations").mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps({"dataset": "Heartbeat"}), encoding="utf-8")
    header = "candidate_hash,status,macro_f1,accuracy,parameters,macs,training_seconds,tokenizer,num_blocks,direction,d_model,d_state,d_conv,expand,pooling\n"
    row = "abc,completed,0.8,0.75,100,200,1.0,point,1,forward,64,8,2,1,mean\n"
    (dataset / "search" / "evaluations.csv").write_text(header + row, encoding="utf-8")
    (dataset / "search" / "pareto_front.csv").write_text(header + row, encoding="utf-8")
    (dataset / "search" / "generations" / "generation_0001.json").write_text(
        json.dumps({"generation": 1, "hypervolume": 0.5}), encoding="utf-8"
    )
    destination = export_paper(run, tmp_path / "paper")
    assert (destination / "all_candidates.csv").exists()
    assert (destination / "search_space_ablation.csv").exists()
    assert not list(destination.rglob("*.pt"))
    assert not list(destination.rglob("*.npz"))

