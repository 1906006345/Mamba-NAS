from __future__ import annotations

import numpy as np


def classification_metrics(targets, predictions) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(targets, predictions)),
    }


def prediction_arrays(logits: np.ndarray, targets: np.ndarray) -> dict:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "predictions": logits.argmax(axis=1),
        "targets": targets,
    }

