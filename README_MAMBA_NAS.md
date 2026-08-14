# MambaTSC-NSGA2

This repository is a complete, source-preserving derivative of
[THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library), distributed under its MIT
license. All upstream entry points remain intact. The new `mamba_nas/` package implements an
independent NSGA-II search pipeline for raw time-series classification with the official Mamba
CUDA implementation.

## Reproducible workflow

Use Ubuntu 22.04 under WSL2. From the repository root:

```bash
# First run scripts/enable_wsl2.ps1 from elevated Windows PowerShell if WSL is not enabled.
bash scripts/setup_mamba_nas_wsl.sh
python -m mamba_nas.cli download --suite uea10
python -m mamba_nas.cli search --dataset Heartbeat --budget smoke
python -m mamba_nas.cli search-all --suite uea10 --budget paper --run-id paper-uea10
python -m mamba_nas.cli refit --run paper-uea10
python -m mamba_nas.cli export-paper --run paper-uea10
```

Long searches accept `--resume`, `--seed`, `--output-dir`, and `--data-dir`. Searches are sequential
on one GPU. A failed candidate is recorded and retried once.

## Search protocol

Each of the 10 UEA datasets receives an independent Pareto front. The 864 legal architectures vary
the tokenizer (point/patch-8/patch-16), 1–3 residual blocks, forward/shared-weight bidirectional
scan, `d_model`, `d_state`, `d_conv`, `expand`, and masked pooling. NSGA-II minimizes
`1 - validation Macro-F1` and `log(1 + analytical MACs)`.

The original TRAIN split is stratified 80/20 with seed 2021. Statistics are fitted only to the inner
training samples. TEST is not loaded during search; it is accessed only after the high-accuracy,
knee, and low-cost Pareto representatives are selected and retrained on complete TRAIN with seeds
2021, 2022, and 2023.

## Artifacts and Git policy

Full checkpoints, curves, predictions, and search state are stored under `runs/` locally. Lightweight
paper tables are regenerated into `paper_exports/`. Raw UEA data, packed environments, `.venv`, and
full run outputs are excluded from Git. The download command records source URLs and file sizes in a
dataset manifest. Analytical MACs consistently count tokenizer projection, Mamba input/output and
state projections, depthwise convolution, selective state update/readout, pooling, and the classifier;
normalization and elementwise activations are excluded. Shared bidirectional blocks count two scans.

Official `mamba_ssm` targets Linux/NVIDIA. This project intentionally does not fall back to the
repository's serial reference Mamba or native Windows CUDA. If `wsl --status` returns access denied,
enable the WSL service from an elevated PowerShell and reboot Windows before running the installer.
