from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np

from .metrics import classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@dataclass
class TrainResult:
    model: object
    history: list[dict]
    metrics: dict
    logits: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    training_seconds: float


def evaluate(model, loader, device: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    logits_all, targets_all = [], []
    with torch.no_grad():
        for values, mask, targets in loader:
            logits = model(values.to(device), mask.to(device))
            logits_all.append(logits.float().cpu().numpy())
            targets_all.append(targets.numpy())
    logits_array = np.concatenate(logits_all)
    targets_array = np.concatenate(targets_all)
    predictions = logits_array.argmax(axis=1)
    return classification_metrics(targets_array, predictions), logits_array, predictions, targets_array


def train_classifier(
    model,
    train_loader,
    validation_loader,
    device: str,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainResult:
    import torch

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_score, best_state, stale = -1.0, None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, mask, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device), mask.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, _, _, _ = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, logits, predictions, targets = evaluate(model, validation_loader, device)
    return TrainResult(
        model=model,
        history=history,
        metrics=metrics,
        logits=logits,
        predictions=predictions,
        targets=targets,
        training_seconds=time.perf_counter() - started,
    )


def train_full_classifier(
    model,
    train_loader,
    evaluation_loader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainResult:
    """Train for a fixed epoch count; TEST is observed only after all updates finish."""
    import torch

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for values, mask, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device), mask.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    metrics, logits, predictions, targets = evaluate(model, evaluation_loader, device)
    return TrainResult(
        model=model,
        history=history,
        metrics=metrics,
        logits=logits,
        predictions=predictions,
        targets=targets,
        training_seconds=time.perf_counter() - started,
    )
