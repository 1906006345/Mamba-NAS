from __future__ import annotations

import logging
import sys
from pathlib import Path


LOGGER_NAME = "mamba_nas"
DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "result.txt"


def configure_logging(log_path: str | Path) -> Path:
    """Route search/refit progress to stdout and a dedicated run log file."""
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Replace only handlers installed here so repeated runs do not duplicate messages.
    for handler in list(logger.handlers):
        if getattr(handler, "_mamba_nas_handler", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    for handler in (stream_handler, file_handler):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        handler._mamba_nas_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    logger.info("%s", "=" * 88)
    logger.info("日志已启用，搜索/重训练过程将写入 %s", path)
    return path


def log_test_result(
    dataset: str,
    role: str,
    seed: int,
    metrics: dict,
    aliased: bool,
    result_path: str | Path = DEFAULT_RESULT_PATH,
) -> None:
    """Append only final TEST metrics to the project-level result file."""
    path = Path(result_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{LOGGER_NAME}.test_result")
    line = (
        "[TEST最终结果] dataset=%s role=%s seed=%d candidate=%s "
        "macro_f1=%.6f accuracy=%.6f parameters=%d macs=%d flops=%s "
        "latency_ms=%s peak_memory=%s seconds=%.2f aliased=%s"
    ) % (
        dataset,
        role,
        seed,
        metrics["candidate_hash"][:12],
        metrics["macro_f1"],
        metrics["accuracy"],
        metrics["parameters"],
        metrics["macs"],
        metrics.get("flops", metrics.get("analytical_flops", "")),
        metrics.get("latency_median_ms", ""),
        metrics.get("peak_memory_bytes", ""),
        metrics["training_seconds"],
        aliased,
    )
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    try:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.info("%s", line)
        handler.flush()
    finally:
        logger.removeHandler(handler)
        handler.close()
