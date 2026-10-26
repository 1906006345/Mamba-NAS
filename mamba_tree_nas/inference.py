"""Load one retained first-seed model, using its full-TRAIN normalization."""
from pathlib import Path
import numpy as np
import torch
from .architecture import Architecture
from .data import Normalizer
from .model import Classifier


def load_retained_model(path, device="cuda"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    arch = Architecture.from_dict(checkpoint["architecture"])
    norm = checkpoint["normalization"]
    normalizer = Normalizer(np.asarray(norm["mean"]), np.asarray(norm["scale"]))
    model = Classifier(arch, len(normalizer.mean), len(checkpoint["classes"]), dropout=0, backend=checkpoint["backend"], device=device).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, normalizer, checkpoint["classes"]
