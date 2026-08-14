from __future__ import annotations

UEA10 = (
    "EthanolConcentration",
    "FaceDetection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "PEMS-SF",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "UWaveGestureLibrary",
)

HF_DATASET_BASE = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
DEFAULT_DATA_DIR = "data/mamba_nas/UEA"
DEFAULT_OUTPUT_DIR = "runs"
SPLIT_SEED = 2021

BUDGETS = {
    "smoke": {
        "population_size": 8,
        "max_unique_candidates": 24,
        "epochs": 3,
        "patience": 3,
        "batch_size": 16,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
    },
    "paper": {
        "population_size": 24,
        "max_unique_candidates": 384,
        "epochs": 20,
        "patience": 3,
        "batch_size": 32,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
    },
}

