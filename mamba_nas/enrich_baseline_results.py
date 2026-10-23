"""Add parameter and forward-MAC estimates to existing fair UEA results.

The script uses the matching fair-protocol checkpoint for exact trainable
parameter counts. MACs are measured with torch.profiler for a batch of one and
reported as FLOPs / 2 (one multiply-add = one MAC). The TimesNet top_k values
are the settings used for the saved runs: Heartbeat/SpokenArabicDigits=1 and
UWaveGestureLibrary=3.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace


DATASET_SHAPES = {
    "Heartbeat": (405, 61, 2),
    "SpokenArabicDigits": (93, 13, 10),
    "UWaveGestureLibrary": (315, 3, 8),
}
TIMESNET_TOP_K = {"Heartbeat": 1, "SpokenArabicDigits": 1, "UWaveGestureLibrary": 3}


def _config(dataset: str, model: str, name: str) -> SimpleNamespace:
    seq_len, channels, classes = DATASET_SHAPES[dataset]
    match = re.search(r"_dm(\d+)_nh\d+_el(\d+)_dl\d+_df(\d+)_", name)
    if not match:
        raise ValueError(f"Could not parse model dimensions from {name}")
    d_model, e_layers, d_ff = map(int, match.groups())
    return SimpleNamespace(
        task_name="classification",
        seq_len=seq_len,
        pred_len=0,
        label_len=48,
        enc_in=channels,
        num_class=classes,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        n_heads=8,
        dropout=0.1,
        activation="gelu",
        factor=1,
        top_k=TIMESNET_TOP_K[dataset],
        num_kernels=6,
        embed="timeF",
        freq="h",
    )


def _profile(model, config: SimpleNamespace) -> int:
    import torch

    model.eval()
    # TimesNet chooses dominant periods from the input spectrum. Use one fixed
    # seeded input so repeated enrichment runs produce the same MAC estimate.
    torch.manual_seed(0)
    values = torch.randn(1, config.seq_len, config.enc_in)
    mask = torch.ones(1, config.seq_len)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    )
    profiler.start()
    with torch.no_grad():
        model(values, mask, None, None)
    profiler.stop()
    flops = sum(event.flops for event in profiler.key_averages())
    return int(flops // 2)


def enrich(results_dir: str | Path = "results", checkpoints_dir: str | Path = "checkpoints") -> None:
    import torch
    from models import PatchTST, TimesNet

    results_dir = Path(results_dir)
    checkpoints_dir = Path(checkpoints_dir)
    aggregate = []
    for result_file in sorted(results_dir.glob("classification_*_*/result_classification.txt")):
        name = result_file.parent.name
        match = re.match(
            r"classification_(Heartbeat|SpokenArabicDigits|UWaveGestureLibrary)_(PatchTST|TimesNet)_",
            name,
        )
        if not match:
            continue
        dataset, model_name = match.groups()
        exact_checkpoint = checkpoints_dir / name / "checkpoint.pth"
        checkpoint_candidates = [exact_checkpoint] if exact_checkpoint.exists() else sorted(
            checkpoints_dir.glob(f"classification_{dataset}_{model_name}_*fair_s2021/checkpoint.pth")
        )
        if not checkpoint_candidates:
            raise FileNotFoundError(f"Missing matching checkpoint for {name}")
        checkpoint = checkpoint_candidates[0]
        config = _config(dataset, model_name, name)
        module = PatchTST if model_name == "PatchTST" else TimesNet
        model = module.Model(config).float()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        macs = _profile(model, config)
        text = result_file.read_text(encoding="utf-8")
        text = re.sub(r"\nparameters: .*\nmacs: .*\n?", "\n", text)
        if not text.endswith("\n"):
            text += "\n"
        text += (
            f"parameters: {parameters}\nmacs: {macs}\nflops: {2 * macs}\n"
            f"profile_macs: {macs}\nprofile_flops: {2 * macs}\n"
            "profile_flops_status: available\n"
        )
        result_file.write_text(text, encoding="utf-8")
        (result_file.parent / "performance_metrics.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "seed": re.search(r"_fair_s(\d+)$", name).group(1),
                    "parameters": parameters,
                    "macs": macs,
                    "flops": 2 * macs,
                    "profile_macs": macs,
                    "profile_flops": 2 * macs,
                    "profile_flops_status": "available",
                    "macs_definition": "torch.profiler forward FLOPs / 2, batch=1, fixed seed=0",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        metrics = {}
        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in {"accuracy", "macro_f1"}:
                    metrics[key.strip()] = float(value.strip())
        aggregate.append(
            {
                "dataset": dataset,
                "model": model_name,
                "seed": re.search(r"_fair_s(\d+)$", name).group(1),
                **metrics,
                "parameters": parameters,
                "macs": macs,
                "flops": 2 * macs,
                "profile_macs": macs,
                "profile_flops": 2 * macs,
                "profile_flops_status": "available",
                "macs_definition": "torch.profiler forward FLOPs / 2, batch=1, fixed seed=0",
            }
        )
        print(f"{dataset} {model_name}: parameters={parameters} macs={macs}")
    aggregate_path = results_dir / "baseline_performance_metrics.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "model", "seed", "accuracy", "macro_f1", "parameters", "macs", "flops", "profile_macs", "profile_flops", "profile_flops_status", "macs_definition"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    print(aggregate_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    args = parser.parse_args()
    enrich(args.results_dir, args.checkpoints_dir)
