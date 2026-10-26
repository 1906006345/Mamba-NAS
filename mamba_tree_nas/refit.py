from __future__ import annotations

import os
from pathlib import Path
import time
import numpy as np
import torch
from .architecture import Architecture
from .artifacts import Events, atomic_json, error_record, implementation_fingerprint, read_json, run_lock
from .config import digest, merge
from .data import DataProtocol
from .profiling import profile_model
from .training import evaluate, train


def _save_weights(path, payload):
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def refit(path, cfg=None, resume=False):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    saved_cfg = manifest["config"]
    if manifest.get("implementation") != implementation_fingerprint():
        raise ValueError("Core implementation changed after search; refit requires the original source revision")
    if cfg is not None and digest(cfg) != digest(saved_cfg):
        raise ValueError("Refit config must match the resolved search config; use the saved manifest")
    cfg = saved_cfg
    state = read_json(path / "search_state.json")
    if state["status"] != "complete":
        raise ValueError("Finish search before refit")
    roles = read_json(path / "representatives.json")
    identity = manifest["identity"]
    data = DataProtocol(cfg, identity["dataset"], identity["synthetic"])
    if data.manifest != manifest["data"]:
        raise ValueError("TRAIN data/split changed since search")
    with run_lock(path / ".refit.lock"):
        output = path / "refit"
        output.mkdir(parents=True, exist_ok=True)
        progress_path = output / "state.json"
        if progress_path.exists() and not resume:
            raise FileExistsError("Refit already started; use --resume")
        progress = read_json(progress_path) if progress_path.exists() else {"status": "running", "completed": [], "current": None}
        if progress["status"] == "complete":
            return output
        events = Events(path / "tensorboard" / "refit", cfg)
        try:
            for h, arch_dict in sorted(roles["architectures"].items()):
                arch = Architecture.from_dict(arch_dict)
                directory = output / h
                directory.mkdir(parents=True, exist_ok=True)
                aliases = sorted(role for role, selected in roles["roles"].items() if selected == h)
                atomic_json(directory / "architecture.json", {"architecture": arch_dict, "hash": h, "aliases": aliases})
                calibration_path = directory / "calibration.json"
                if not calibration_path.exists():
                    progress.update(status="running", current={"hash": h, "phase": "calibration"})
                    atomic_json(progress_path, progress)
                    calibration_cfg = merge(cfg, {"train": cfg["calibration"]})
                    result = train(arch, data, calibration_cfg, cfg["seeds"]["calibration"], epoch_callback=lambda row: _log(events, f"{h[:12]}/calibration", row))
                    atomic_json(calibration_path, {k: result[k] for k in ("best_epoch", "epochs_run", "validation", "training_seconds", "history")})
                    del result
                calibration = read_json(calibration_path)
                full_train, normalizer = data.full_train()
                for i, seed in enumerate(cfg["seeds"]["refit"]):
                    result_path = directory / f"seed_{seed}.json"
                    key = f"{h}:{seed}"
                    if result_path.exists():
                        if key not in progress["completed"]:
                            progress["completed"].append(key)
                        continue
                    progress.update(status="running", current={"hash": h, "phase": "refit", "seed": seed})
                    atomic_json(progress_path, progress)
                    result = train(arch, data, cfg, seed, train_data=full_train, fixed_epochs=calibration["best_epoch"], epoch_callback=lambda row: _log(events, f"{h[:12]}/seed_{seed}", row))
                    # TEST is loaded only after all optimizer updates for this fit.
                    test = data.load_test(normalizer)
                    metrics, predictions = evaluate(result["model"], test, cfg, len(data.classes))
                    profile = profile_model(result["model"], full_train, cfg)
                    payload = {"hash": h, "aliases": aliases, "refit_seed": seed, "epochs": calibration["best_epoch"], "calibration_seed": cfg["seeds"]["calibration"], "training_seconds": result["training_seconds"], "training_peak_allocated_gpu_bytes": result["training_peak_allocated_gpu_bytes"], "parameters": result["parameters"], "history": result["history"], "test": metrics, "profile": profile, "normalization": normalizer.to_dict(), "synthetic": identity["synthetic"], "backend": cfg["backend"]}
                    if cfg["retention"]["predictions"]:
                        temp = directory / f"seed_{seed}_predictions.tmp.npz"
                        np.savez_compressed(temp, **predictions)
                        os.replace(temp, directory / f"seed_{seed}_predictions.npz")
                    if i == 0 and cfg["retention"]["save_first_refit_weights"]:
                        _save_weights(directory / "first_seed.pt", {"architecture": arch_dict, "state_dict": {k: v.cpu() for k, v in result["model"].state_dict().items()}, "normalization": normalizer.to_dict(), "classes": data.classes, "seed": seed, "backend": cfg["backend"], "protocol_fingerprint": manifest["fingerprint"]})
                    # Result is the commit marker, written after all required artifacts.
                    atomic_json(result_path, payload)
                    progress["completed"].append(key)
                    atomic_json(progress_path, progress)
                    del result, predictions, test
                    if str(cfg["device"]).startswith("cuda"):
                        torch.cuda.empty_cache()
            progress.update(status="complete", current=None)
            atomic_json(progress_path, progress)
            return output
        except BaseException as exc:
            progress.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", last_error=error_record(exc, "refit"))
            atomic_json(progress_path, progress)
            raise
        finally:
            events.close()


def _log(events, prefix, row):
    for key, value in row.items():
        if key != "epoch":
            events.scalar(prefix + "/" + key, value, row["epoch"])
