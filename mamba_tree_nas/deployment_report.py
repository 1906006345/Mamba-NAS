"""Read-only status and reproducible CSV/figure exports for a deployment study."""
import csv
import statistics
from collections import defaultdict

from .artifacts import read_json, write_csv
from .deployment_metrics import curves_at, hv_at
from .deployment_search import ARMS, SUPERNET, _history
from .followup_evaluation import atomic_json


def _saved(path):
    return read_json(path) if path.exists() else None


def collect(output):
    protocol = _saved(output / "protocol.json")
    if protocol is None:
        return {"status": "not_started", "datasets": [], "runs": []}, {}, {}
    settings = protocol["config"]["deployment_nas"]
    budget = settings["search"]["budget"]
    report = {"status": "available", "runtime": protocol["runtime"]["mode"], "profile": protocol["config"]["profile"],
              "datasets": [], "runs": []}
    online, audited = {}, {}
    for dataset in settings["datasets"]:
        root = output / dataset
        if not root.exists():
            continue
        benchmark = _saved(root / "benchmark" / "summary.json")
        rank = _saved(root / "rank" / "summary.json")
        report["datasets"].append({"dataset": dataset, "benchmark": benchmark, "rank": rank})
        for seed in settings["search_seeds"]:
            for arm in (*ARMS, SUPERNET):
                path = root / "search" / arm / str(seed)
                state = _saved(path / "state.json")
                pretraining = _saved(root / "supernet" / str(seed) / "summary.json") if arm == SUPERNET else None
                if state is None and pretraining is None:
                    continue
                state = state or {"records": [], "status": "pretrained"}
                start = state.get("pretraining_seconds", pretraining["elapsed_seconds"] if pretraining else 0.)
                key = (dataset, seed, arm)
                history = _history(state["records"], settings["hv_reference"], start, start)
                audit = _saved(path / "audit.json")
                online[key] = history
                audited[key] = audit["curve"] if audit else []
                summary = {"dataset": dataset, "seed": seed, "arm": arm, "search_status": state["status"],
                           "candidates": len(history), "budget": budget, "pretraining_seconds": start,
                           "failed_candidates": sum(r["status"] != "complete" for r in state["records"]),
                           "audit_status": ("complete" if audit and audit["curve"][-1]["failed_audits"] == 0
                                            else "partial" if audit else "not_run")}
                if history:
                    last = history[-1]
                    summary.update(final_common_hv=last["hv"], final_native_hv=last["native_hv"],
                                   final_projected_hv=last["projected_hv"], outside_reference=last["outside_reference"],
                                   wall_seconds=last["wall_seconds"], attributed_seconds=last["attributed_seconds"],
                                   online_hv_auc=curves_at(history, "candidates", last["candidates"]))
                if audit:
                    last_audit = audit["curve"][-1]
                    summary.update(final_audited_hv=last_audit["hv"], audited=last_audit["audited"],
                                   successful_audits=last_audit["successful_audits"],
                                   offline_audit_attributed_seconds=last_audit["audit_attributed_seconds"],
                                   offline_audit_new_fit_seconds=last_audit["audit_new_fit_seconds"],
                                   offline_audit_wall_seconds=audit.get("elapsed_seconds"))
                if state["status"] in ("failed", "interrupted"):
                    summary["last_error"] = state.get("last_error")
                report["runs"].append(summary)
    return report, online, audited


def pair_metrics(report, online, audited):
    complete = {(r["dataset"], r["seed"], r["arm"]): r for r in report["runs"] if r["search_status"] == "complete"}
    result = []
    for dataset, seed in sorted({key[:2] for key in complete}):
        for left, right, phase in (
            (ARMS[0], ARMS[1], "objectives"), (ARMS[1], ARMS[2], "objectives"), (ARMS[2], SUPERNET, "compare"),
        ):
            a, b = (dataset, seed, left), (dataset, seed, right)
            if a not in complete or b not in complete:
                continue
            histories = (audited[a], audited[b]) if phase == "compare" else (online[a], online[b])
            if not all(histories):
                continue
            x, y = histories
            row = {"dataset": dataset, "seed": seed, "baseline": left, "method": right, "stage": phase,
                   "curve": "audited" if phase == "compare" else "online_common",
                   "final_hv_delta": y[-1]["hv"] - x[-1]["hv"],
                   "projected_final_hv_delta": y[-1]["projected_hv"] - x[-1]["projected_hv"],
                   "candidate_hv_auc_delta": curves_at(y, "candidates", y[-1]["candidates"])
                   - curves_at(x, "candidates", x[-1]["candidates"])}
            if phase == "compare":
                row["audit_complete"] = all(complete[k]["audit_status"] == "complete" for k in (a, b))
                row["baseline_audited"] = x[-1]["successful_audits"]
                row["method_audited"] = y[-1]["successful_audits"]
            for key in ("wall_seconds", "attributed_seconds"):
                limit = min(x[-1][key], y[-1][key])
                row["common_" + key] = limit
                row[key + "_final_hv_delta"] = hv_at(y, key, limit) - hv_at(x, key, limit)
                row[key + "_hv_auc_delta"] = curves_at(y, key, limit) - curves_at(x, key, limit)
            result.append(row)
    return result


def _pair_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("audit_complete", True):
            groups[(row["dataset"], row["stage"], row["baseline"], row["method"])].append(row)
    result = []
    for (dataset, stage, baseline, method), members in sorted(groups.items()):
        row = {"dataset": dataset, "stage": stage, "baseline": baseline, "method": method, "paired_seeds": len(members)}
        for key in members[0]:
            if key.endswith("_delta"):
                values = [m[key] for m in members]
                row[key + "_mean"] = statistics.mean(values)
                row[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(row)
    return result


def _plots(output, online, audited):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    for root in sorted(p for p in output.iterdir() if p.is_dir() and p.name != "report"):
        folder = output / "report" / root.name
        for stage, xfield, yfield, xlabel, ylabel in (
            ("rank", "short_f1", "reference_f1", "Short training rank", "Full training rank"),
            ("benchmark", "macs", "latency_ms", "Analytic MACs (max TRAIN length)", "Batch=1 wall-clock P50 (ms)"),
        ):
            path = root / stage / "architectures.csv"
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = [r for r in csv.DictReader(handle) if r.get(xfield) and r.get(yfield)
                        and (stage != "benchmark" or r.get("panel") == "True")]
            if not rows:
                continue
            x, y = ([float(r[key]) for r in rows] for key in (xfield, yfield))
            if stage == "rank":
                x, y = rankdata([-v for v in x]), rankdata([-v for v in y])
            fig, ax = plt.subplots(figsize=(5.5, 4.3), layout="constrained")
            ax.scatter(x, y, color="#235789", alpha=.75)
            ax.set(xlabel=xlabel, ylabel=ylabel, title=root.name)
            if stage == "rank":
                ax.plot([1, len(x)], [1, len(x)], color="#8a8a8a", linestyle="--", linewidth=1)
            else:
                ax.set_xscale("log")
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f"{stage}.png", dpi=160)
            plt.close(fig)

    for label, curves in (("online", online), ("audited", audited)):
        for dataset, seed in sorted({key[:2] for key in curves}):
            entries = [(arm, rows) for (name, s, arm), rows in curves.items() if (name, s) == (dataset, seed) and rows]
            if not entries:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
            for ax, x, axis in zip(axes, ("candidates", "wall_seconds", "attributed_seconds"),
                                   ("Candidates", "Wall-clock seconds", "Attributed seconds")):
                for arm, rows in entries:
                    ax.step([0] + [r[x] for r in rows], [0] + [r["hv"] for r in rows], where="post", label=arm)
                ax.set(xlabel=axis, ylabel="Common 3D HV")
            axes[0].legend(fontsize=7)
            fig.suptitle(f"{dataset} | seed {seed} | {label}")
            destination = output / "report" / dataset
            destination.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination / f"{label}_hv_{seed}.png", dpi=160)
            plt.close(fig)

    for dataset, seed in sorted({key[:2] for key in online}):
        fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
        for arm in ARMS:
            path = output / dataset / "search" / arm / str(seed) / "f1_latency_front.csv"
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8-sig") as handle:
                points = sorted({(float(r["latency_ms"]), 100 * float(r["validation_f1"])) for r in csv.DictReader(handle)})
            if points:
                ax.plot(*zip(*points), marker="o", label=arm)
        if ax.lines:
            ax.set(xlabel="Batch=1 wall-clock P50 (ms)", ylabel="Validation macro-F1 (%)",
                   title=f"{dataset} | seed {seed} | objective comparison")
            ax.legend(fontsize=8)
            destination = output / "report" / dataset
            destination.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination / f"f1_latency_{seed}.png", dpi=160)
        plt.close(fig)


def write_report(output, report, online, audited):
    folder = output / "report"
    for label, curves in (("online", online), ("audited", audited)):
        write_csv(folder / f"{label}_curves.csv", [
            dict(row, dataset=dataset, seed=seed, arm=arm)
            for (dataset, seed, arm), rows in curves.items() for row in rows
        ])
    pairs = pair_metrics(report, online, audited)
    write_csv(folder / "runs.csv", report["runs"])
    write_csv(folder / "paired_deltas.csv", pairs)
    write_csv(folder / "paired_summary.csv", _pair_summary(pairs))
    atomic_json(folder / "status.json", report)
    _plots(output, online, audited)
