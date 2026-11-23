"""Tree-only ranking diagnosis and shared fully trained reference scores."""
import statistics

from .artifacts import write_csv
from .deployment_metrics import rank_summary
from .followup_evaluation import atomic_json


def score_seeds(study, arch, mode):
    results = []
    for seed in study.settings["training_seeds"]:
        try:
            result, hit = study.standalone(arch, seed, mode)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            results.append({"seed": seed, "status": "failed", "error": str(exc)})
        else:
            results.append({"seed": seed, "status": "complete", "f1": result["validation"]["macro_f1"],
                            "epochs_run": result["epochs_run"], "best_epoch": result["best_epoch"],
                            "cached": hit, "elapsed_seconds": result["elapsed_seconds"]})
    return results


def summarize_seeds(results):
    scores = [r["f1"] for r in results if r["status"] == "complete"]
    if len(scores) != len(results):
        return None, None
    return statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else None


def rank(study):
    rows, fits = [], []
    for arch in study.panel:
        row = {"hash": arch.hash, "architecture": arch.to_dict()}
        for mode in ("short", "reference"):
            results = score_seeds(study, arch, mode)
            row[mode + "_f1"], row[mode + "_seed_sd"] = summarize_seeds(results)
            fits.extend(dict(r, hash=arch.hash, protocol=mode) for r in results)
        rows.append(row)
        print(f"[RANK] {study.data.name} {len(rows)}/{len(study.panel)} {arch.hash[:8]}", flush=True)
    result = rank_summary(rows)
    result["status"] = "complete" if result["failed_pairs"] == 0 else "partial"
    result["training_seeds"] = study.settings["training_seeds"]
    output = study.root / "rank"
    write_csv(output / "architectures.csv", rows)
    write_csv(output / "fits.csv", fits)
    write_csv(output / "summary.csv", [result])
    atomic_json(output / "summary.json", result)
    return result
