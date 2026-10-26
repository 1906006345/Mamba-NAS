from __future__ import annotations

import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback
from contextlib import contextmanager
from .config import digest


def implementation_fingerprint():
    import hashlib
    from . import __version__
    source = Path(__file__).parent
    core = ("architecture.py", "model.py", "data.py", "cost.py", "metrics.py", "training.py", "proxy.py", "search.py", "refit.py", "profiling.py", "config.py", "artifacts.py")
    return {"version": __version__, "source_sha256": digest({name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in core})}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def environment():
    packages = {}
    for name in ("torch", "mamba-ssm", "causal-conv1d", "numpy", "scipy", "scikit-learn", "PyYAML", "matplotlib", "tensorboard"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    except ImportError:
        result["cuda_available"] = False
    return result


def error_record(exc, phase):
    return {"phase": phase, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "time_unix": time.time()}


class Events:
    def __init__(self, path, cfg, purge_step=None):
        self.writer = None
        if cfg["logging"]["tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(path), flush_secs=cfg["logging"]["flush_seconds"], purge_step=purge_step)

    def scalar(self, name, value, step):
        if self.writer and value is not None:
            self.writer.add_scalar(name, value, step)

    def close(self):
        if self.writer:
            self.writer.close()


@contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # OS lock releases on process death; the empty file is safe after crashes.
    handle = path.open("a+b")
    handle.seek(0); handle.write(b"0"); handle.flush(); handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Run is already active: {path.parent}") from exc
    try:
        yield
    finally:
        handle.close()


def init_run(path, cfg, identity, data_manifest, resume):
    path = Path(path)
    manifest = {"config": cfg, "identity": identity, "data": data_manifest, "implementation": implementation_fingerprint()}
    fingerprint = digest(manifest)
    config_path = path / "manifest.json"
    if config_path.exists():
        old = read_json(config_path)
        if not resume:
            raise FileExistsError(f"Run exists; use --resume: {path}")
        if old["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: resolved config, identity or data protocol changed")
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume absent run: {path}")
        atomic_json(config_path, dict(manifest, fingerprint=fingerprint))
        atomic_json(path / "environment.json", environment())
        atomic_json(path / "split.json", data_manifest)
    return fingerprint
