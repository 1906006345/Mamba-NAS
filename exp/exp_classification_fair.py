"""Leakage-free optional classification protocol.

This module deliberately lives beside the legacy experiment instead of replacing it.
It uses only TRAIN for model/epoch selection, refits on complete TRAIN, and loads TEST
once after every parameter update has finished.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

from data_provider.uea import collate_fn
from exp.exp_basic import Exp_Basic
from mamba_nas.data import (
    PaddedSeriesDataset,
    fit_normalization,
    load_ts,
    normalize_samples,
)
from mamba_nas.performance import model_performance


class Exp_Classification_Fair(Exp_Basic):
    """Optional TRAIN/validation/refit/TEST protocol for UEA classification."""

    def __init__(self, args):
        if args.data != "UEA":
            raise ValueError("--classification_protocol fair currently supports --data UEA only")
        self.train_collection = None
        self.full_train_stats = None
        self.best_epoch = None
        self.inner_history = []
        self.refit_history = []
        super().__init__(args)

    @property
    def _train_path(self) -> Path:
        return Path(self.args.root_path) / f"{self.args.model_id}_TRAIN.ts"

    @property
    def _test_path(self) -> Path:
        return Path(self.args.root_path) / f"{self.args.model_id}_TEST.ts"

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _build_model(self):
        # Only TRAIN is loaded here. In particular, seq_len and class names do not
        # inspect TEST metadata.
        self._seed_everything(self.args.classification_seed)
        self.train_collection = load_ts(self._train_path)
        self.args.seq_len = self.train_collection.max_length
        self.args.pred_len = 0
        self.args.enc_in = self.train_collection.input_channels
        self.args.num_class = len(self.train_collection.class_names)
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _make_loader(self, samples, labels, shuffle: bool, seed: int) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            PaddedSeriesDataset(samples, labels),
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: collate_fn(batch, max_len=self.args.seq_len),
        )

    def _new_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    @staticmethod
    def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
        return {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        }

    def _evaluate(self, loader, criterion) -> dict:
        self.model.eval()
        losses, logits_all, targets_all = [], [], []
        with torch.no_grad():
            for batch_x, labels, padding_mask in loader:
                outputs = self.model(
                    batch_x.float().to(self.device),
                    padding_mask.float().to(self.device),
                    None,
                    None,
                )
                targets = labels.long().squeeze(-1).to(self.device)
                losses.append(float(criterion(outputs, targets).detach().cpu()))
                logits_all.append(outputs.float().cpu().numpy())
                targets_all.append(targets.cpu().numpy())
        logits = np.concatenate(logits_all)
        targets = np.concatenate(targets_all).reshape(-1)
        predictions = logits.argmax(axis=1)
        self.model.train()
        return {
            "loss": float(np.mean(losses)),
            **self._metrics(targets, predictions),
            "logits": logits,
            "predictions": predictions,
            "targets": targets,
        }

    def _train_epoch(self, loader, optimizer, criterion) -> float:
        self.model.train()
        losses = []
        for batch_x, labels, padding_mask in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(
                batch_x.float().to(self.device),
                padding_mask.float().to(self.device),
                None,
                None,
            )
            targets = labels.long().squeeze(-1).to(self.device)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def _fair_setting(self, setting: str) -> str:
        return f"{setting}_fair_s{self.args.classification_seed}"

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0]) if rows else []
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def train(self, setting):
        fair_setting = self._fair_setting(setting)
        checkpoint_dir = Path(self.args.checkpoints) / fair_setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        seed = self.args.classification_seed
        labels = self.train_collection.labels
        indices = np.arange(len(labels))
        _, class_counts = np.unique(labels, return_counts=True)
        if np.any(class_counts < 2):
            raise ValueError("Fair stratified split requires at least two TRAIN samples per class")

        inner_indices, validation_indices = train_test_split(
            indices,
            test_size=self.args.classification_val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        inner_indices = np.sort(inner_indices)
        validation_indices = np.sort(validation_indices)
        np.savez_compressed(
            checkpoint_dir / "split_indices.npz",
            inner_train=inner_indices,
            validation=validation_indices,
        )

        inner_raw = [self.train_collection.samples[i] for i in inner_indices]
        validation_raw = [self.train_collection.samples[i] for i in validation_indices]
        inner_stats = fit_normalization(inner_raw)
        inner_samples = normalize_samples(inner_raw, inner_stats)
        validation_samples = normalize_samples(validation_raw, inner_stats)
        np.savez_compressed(
            checkpoint_dir / "inner_normalization_stats.npz",
            mean=inner_stats.mean,
            std=inner_stats.std,
        )
        train_loader = self._make_loader(inner_samples, labels[inner_indices], True, seed)
        validation_loader = self._make_loader(
            validation_samples, labels[validation_indices], False, seed
        )

        optimizer = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_score = -np.inf
        best_state = None
        best_epoch = 0
        stale = 0
        self.inner_history = []

        for epoch in range(1, self.args.train_epochs + 1):
            started = time.perf_counter()
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            validation = self._evaluate(validation_loader, criterion)
            score = validation[self.args.classification_monitor]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "seconds": time.perf_counter() - started,
            }
            self.inner_history.append(row)
            print(
                "Epoch: {epoch} | Train Loss: {train_loss:.4f} | "
                "Val Loss: {val_loss:.4f} Val Acc: {accuracy:.4f} "
                "Val Macro-F1: {macro_f1:.4f}".format(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=validation["loss"],
                    accuracy=validation["accuracy"],
                    macro_f1=validation["macro_f1"],
                )
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, checkpoint_dir / "inner_best_checkpoint.pth")
                stale = 0
            else:
                stale += 1
                if stale >= self.args.patience:
                    print("Early stopping on TRAIN-derived validation")
                    break

        if best_state is None:
            raise RuntimeError("Fair classification did not produce a validation checkpoint")
        self.best_epoch = best_epoch
        self._write_csv(checkpoint_dir / "inner_history.csv", self.inner_history)

        # Refit from a fresh initialization on complete TRAIN for the selected
        # epoch count. No TEST loader exists at this point.
        self._seed_everything(seed)
        self.model = self._new_model()
        self.full_train_stats = fit_normalization(self.train_collection.samples)
        full_train_samples = normalize_samples(
            self.train_collection.samples, self.full_train_stats
        )
        np.savez_compressed(
            checkpoint_dir / "full_train_normalization_stats.npz",
            mean=self.full_train_stats.mean,
            std=self.full_train_stats.std,
        )
        full_train_loader = self._make_loader(full_train_samples, labels, True, seed)
        optimizer = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        self.refit_history = []
        for epoch in range(1, best_epoch + 1):
            started = time.perf_counter()
            train_loss = self._train_epoch(full_train_loader, optimizer, criterion)
            self.refit_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "seconds": time.perf_counter() - started,
                }
            )
            print(f"Refit epoch: {epoch}/{best_epoch} | Train Loss: {train_loss:.4f}")
        torch.save(self.model.state_dict(), checkpoint_dir / "checkpoint.pth")
        self._write_csv(checkpoint_dir / "refit_history.csv", self.refit_history)
        with (checkpoint_dir / "protocol.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol": "fair",
                    "test_accessed_during_training": False,
                    "selection_split": "stratified TRAIN inner/validation",
                    "validation_ratio": self.args.classification_val_ratio,
                    "monitor": self.args.classification_monitor,
                    "best_epoch": best_epoch,
                    "seed": seed,
                    "normalization": "inner TRAIN stats for selection; full TRAIN stats for refit/TEST",
                    "sequence_length_source": "TRAIN only",
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Selected epoch {best_epoch}; refit completed without accessing TEST")
        return self.model

    def test(self, setting, test=0):
        fair_setting = self._fair_setting(setting)
        checkpoint_dir = Path(self.args.checkpoints) / fair_setting
        if test:
            self.model.load_state_dict(
                torch.load(checkpoint_dir / "checkpoint.pth", map_location=self.device)
            )
        if self.full_train_stats is None:
            self.full_train_stats = fit_normalization(self.train_collection.samples)

        # This is the first TEST load in a normal fair train -> test run.
        test_collection = load_ts(
            self._test_path,
            class_names=self.train_collection.class_names,
        )
        test_samples = normalize_samples(test_collection.samples, self.full_train_stats)
        test_loader = self._make_loader(
            test_samples,
            test_collection.labels,
            False,
            self.args.classification_seed,
        )
        criterion = nn.CrossEntropyLoss()
        result = self._evaluate(test_loader, criterion)
        performance = model_performance(self.model, test_loader, self.device)
        print(f"test accuracy: {result['accuracy']:.6f}")
        print(f"test macro_f1: {result['macro_f1']:.6f}")
        print(f"parameters: {performance['parameters']}")
        print(f"macs: {performance['macs']}")
        print(f"flops: {performance.get('flops')}")
        print(f"profile_macs: {performance.get('profile_macs')}")
        print(f"profile_flops: {performance.get('profile_flops')}")
        print(f"latency_median_ms: {performance.get('latency_median_ms')}")
        print(f"latency_p90_ms: {performance.get('latency_p90_ms')}")
        print(f"throughput_samples_per_sec: {performance.get('throughput_samples_per_sec')}")
        print(f"peak_memory_bytes: {performance.get('peak_memory_bytes')}")

        output_dir = Path("results") / fair_setting
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "result_classification.txt").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(f"{fair_setting}\n")
            handle.write(f"accuracy: {result['accuracy']:.6f}\n")
            handle.write(f"macro_f1: {result['macro_f1']:.6f}\n\n")
            handle.write(f"parameters: {performance['parameters']}\n")
            handle.write(f"macs: {performance['macs']}\n")
            handle.write(f"macs_definition: {performance['macs_definition']}\n")
            handle.write(f"flops: {performance.get('flops')}\n")
            handle.write(f"profile_macs: {performance.get('profile_macs')}\n")
            handle.write(f"profile_flops: {performance.get('profile_flops')}\n")
            handle.write(f"latency_median_ms: {performance.get('latency_median_ms')}\n")
            handle.write(f"latency_p90_ms: {performance.get('latency_p90_ms')}\n")
            handle.write(f"throughput_samples_per_sec: {performance.get('throughput_samples_per_sec')}\n")
            handle.write(f"peak_memory_bytes: {performance.get('peak_memory_bytes')}\n")
            handle.write(f"efficiency_definition: {performance.get('efficiency_definition')}\n\n")
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            logits=result["logits"],
            predictions=result["predictions"],
            targets=result["targets"],
        )
        report = classification_report(
            result["targets"],
            result["predictions"],
            labels=np.arange(len(self.train_collection.class_names)),
            target_names=self.train_collection.class_names,
            output_dict=True,
            zero_division=0,
        )
        with (output_dir / "classification_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        matrix = confusion_matrix(
            result["targets"],
            result["predictions"],
            labels=np.arange(len(self.train_collection.class_names)),
        )
        np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
        return result
