from __future__ import annotations

import csv
import math
from pathlib import Path
import numpy as np
from .architecture import Architecture, at, paths
from .artifacts import atomic_json, read_json, write_csv
from .config import digest, protocol_fingerprint
from .metrics import nested_summary, pareto, proxy_diagnostics
from .search import load_records


def _figure(fig, path, dpi, synthetic=False):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, .09 if synthetic else 0, 1, 1))
    if synthetic:
        fig.text(.5, .005, "SYNTHETIC / TEST MIXER - plumbing verification only", ha="center", fontsize=8, color="#a33220")
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def architecture_figure(arch, path, dpi, synthetic=False):
    import matplotlib.pyplot as plt
    coordinates, counter = {}, [0]
    def layout(node, key=(), depth=0):
        if not node.children:
            x = counter[0]; counter[0] += 1
        else:
            child_x = [layout(c, key + (i,), depth + 1) for i, c in enumerate(node.children)]
            x = sum(child_x) / len(child_x)
        coordinates[key] = (x, -depth)
        return x
    layout(arch.tree)
    fig, ax = plt.subplots(figsize=(max(5, arch.tree.leaves * 2.6), max(3, arch.tree.depth * 1.2 + 2)))
    for key in paths(arch.tree):
        node = at(arch.tree, key)
        x, y = coordinates[key]
        if key:
            px, py = coordinates[key[:-1]]
            ax.plot([px, x], [py, y], color="0.6", zorder=1)
            if at(arch.tree, key[:-1]).op == "SEQ":
                ax.text((px+x)/2, (py+y)/2, "F" if key[-1] == 0 else "G", fontsize=8)
        label = node.op if node.leaf is None else f"OP {node.leaf.direction}\ns={node.leaf.d_state}, c={node.leaf.d_conv}, e={node.leaf.expand}"
        ax.text(x, y, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.4", "fc": "#e9f0fa", "ec": "#396897"}, fontsize=9, zorder=2)
    ax.set_title(f"{arch.tokenizer}, D={arch.d_model}, {arch.pooling}\n{arch.hash[:12]}")
    ax.set_xlim(-.9, max(.9, arch.tree.leaves - .1)); ax.set_ylim(-arch.tree.depth - .7, .9); ax.axis("off")
    _figure(fig, path, dpi, synthetic)


def hv_auc(history, coordinate):
    x = [0.0] + [r[coordinate] for r in history]
    y = [0.0] + [r["hv"] for r in history]
    area = sum(y[i] * (x[i+1] - x[i]) for i in range(len(x) - 1))
    return {"auc": area, "mean_hv": area / x[-1] if x[-1] else 0.0, "integration": "right-continuous step HV, starting at (0,0)"}


def _run_figures(directory, records, state, roles, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = cfg["report"]["dpi"]
    good = [r for r in records if r["status"] == "complete"]
    if not good:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r["macs"] for r in good], [r["validation"]["macro_f1"] for r in good], alpha=.45, label="Evaluated")
    front = sorted(pareto(records), key=lambda r: r["macs"])
    ax.plot([r["macs"] for r in front], [r["validation"]["macro_f1"] for r in front], "o-", label="All-time Pareto")
    ax.set(xscale="log", xlabel="Analytic MACs (max TRAIN length)", ylabel="Validation Macro-F1"); ax.legend(); ax.grid(alpha=.2)
    _figure(fig, directory / "pareto", dpi, cfg["backend"] == "test")
    history = state["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, label in zip(axes, ("candidate_count", "elapsed_seconds"), ("Unique candidates (including failures)", "Cumulative search seconds")):
        ax.step([0] + [r[key] for r in history], [0] + [r["hv"] for r in history], where="post")
        ax.set(xlabel=label, ylabel="Fixed-bound HV"); ax.grid(alpha=.2)
    _figure(fig, directory / "hv", dpi, cfg["backend"] == "test")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for h in sorted(set(roles.values())):
        r = next(r for r in good if r["hash"] == h)
        epochs = r["history"]
        axes[0].plot([x["epoch"] for x in epochs], [x["train_ce"] for x in epochs], label=h[:8] + " train")
        axes[0].plot([x["epoch"] for x in epochs], [x["validation_ce"] for x in epochs], "--", label=h[:8] + " val")
        axes[1].plot([x["epoch"] for x in epochs], [x["validation_macro_f1"] for x in epochs], label=h[:8])
        architecture_figure(Architecture.from_dict(r["architecture"]), directory / ("architecture_" + h[:12]), dpi, cfg["backend"] == "test")
    axes[0].set(xlabel="Epoch", ylabel="CE"); axes[1].set(xlabel="Epoch", ylabel="Validation Macro-F1")
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    _figure(fig, directory / "candidate_training", dpi, cfg["backend"] == "test")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for scope, ax in zip(("full", "trunk"), axes):
        usable = [(r["proxy"].get("aggregate", {}).get(scope, {}).get("log10_mean_product"), r) for r in good]
        usable = [(v, r) for v, r in usable if v is not None]
        if usable:
            points = ax.scatter([v for v, r in usable], [r["validation"]["macro_f1"] for v, r in usable], c=[math.log1p(r["macs"]) for v, r in usable], cmap="viridis")
            fig.colorbar(points, ax=ax, label="log1p(MACs)")
        else:
            ax.text(.5, .5, "No valid proxy observations", ha="center", transform=ax.transAxes)
        ax.set(xlabel=f"{scope}: log10(mean product)", ylabel="Validation Macro-F1"); ax.grid(alpha=.2)
    _figure(fig, directory / "proxy_diagnostic", dpi, cfg["backend"] == "test")


def external_baselines(path, valid_protocols):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "method", "source", "provenance", "protocol_description"}
    for r in rows:
        if not required.issubset(r) or any(not r[k].strip() for k in required):
            raise ValueError("External table needs dataset,method,source,provenance,protocol_description")
        if r["provenance"] not in ("same_protocol", "literature"):
            raise ValueError("External provenance must be same_protocol or literature")
        if r["provenance"] == "same_protocol" and r.get("evaluation_fingerprint") not in valid_protocols.get(r["dataset"], set()):
            raise ValueError("same_protocol baseline must carry a matching exported evaluation_fingerprint")
        scores = [r.get(k, "") for k in ("accuracy", "macro_f1") if r.get(k, "") != ""]
        if not scores or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in scores):
            raise ValueError("External baseline requires finite Accuracy and/or Macro-F1 in [0,1]")
        r["comparability"] = "matched declared protocol" if r["provenance"] == "same_protocol" else "literature result; direct comparability not established"
    return rows


def report(root, output, external=None, *, run_paths=None):
    root, output = Path(root), Path(output)
    manifests = ([Path(p) / "manifest.json" for p in run_paths] if run_paths is not None
                 else [root / "manifest.json"] if (root / "manifest.json").exists()
                 else sorted(root.rglob("manifest.json")))
    if not manifests:
        raise FileNotFoundError(f"No runs under {root}")
    for file in manifests:
        if not file.exists():
            raise FileNotFoundError(f"Run manifest not found: {file}")
    # Validate compatibility before writing any export artifacts.
    preflight = {}
    modes_before = set()
    seen_searches = set()
    for file in manifests:
        m = read_json(file)
        c, identity = m["config"], m["identity"]
        search_key = (identity["dataset"], identity["variant"], identity["search_seed"])
        if search_key in seen_searches:
            raise ValueError("Duplicate run-ids for the same search seed; select one run root explicitly")
        seen_searches.add(search_key)
        modes_before.add((c["backend"], identity["synthetic"]))
        fp = digest({"protocol": protocol_fingerprint(c), "data": digest(m["data"]), "implementation": m.get("implementation")})
        preflight.setdefault(identity["dataset"], set()).add(fp)
    if len(modes_before) > 1 or any(len(p) > 1 for p in preflight.values()):
        raise ValueError("Report aggregation refuses mixed backend/data/evaluation protocols; export each protocol separately")
    rows, search_rows, candidate_rows, epoch_rows, diagnostics, plot_runs = [], [], [], [], [], []
    valid_protocols = {}
    protocols_by_dataset = {}
    modes = set()
    for manifest_path in manifests:
        path = manifest_path.parent
        if not (path / "search_state.json").exists():
            continue
        manifest = read_json(manifest_path)
        cfg, identity = manifest["config"], manifest["identity"]
        modes.add((cfg["backend"], identity["synthetic"]))
        state = read_json(path / "search_state.json")
        records = load_records(path)
        plot_runs.append((identity, state, records, cfg))
        evaluation_fp = digest({"protocol": protocol_fingerprint(cfg), "data": digest(manifest["data"]), "implementation": manifest.get("implementation")})
        valid_protocols.setdefault(identity["dataset"], set()).add(evaluation_fp)
        protocols_by_dataset.setdefault(identity["dataset"], set()).add(evaluation_fp)
        roles = read_json(path / "representatives.json")["roles"] if (path / "representatives.json").exists() else {}
        tag = f"{identity['dataset']}/{identity['variant']}/seed_{identity['search_seed']}/{identity['run_id']}"
        directory = output / tag
        _run_figures(directory, records, state, roles, cfg)
        row = dict(identity, status=state["status"], evaluation_fingerprint=evaluation_fp, backend=cfg["backend"], unique_candidates=len(records), failures=sum(r["status"] == "failed" for r in records), proxy_failures=sum(r["proxy"]["status"] == "failed" for r in records), search_seconds=state["elapsed_seconds"], proxy_seconds=sum(r["proxy"]["seconds"] for r in records), training_seconds=sum(r.get("training_seconds", 0) for r in records), cost_upper_bound=state["cost_bound"], final_hv=state["history"][-1]["hv"] if state["history"] else 0.0)
        for axis in ("candidate_count", "elapsed_seconds"):
            row.update({axis + "_hv_" + k: v for k, v in hv_auc(state["history"], axis).items()})
        if state["history"]:
            row.update({k: state["history"][-1][k] for k in ("pareto_size", "failure_rate", "topology_count", "topology_entropy")})
        search_rows.append(row)
        write_csv(directory / "search_history.csv", state["history"])
        for r in records:
            candidate_rows.append(dict(identity, hash=r["hash"], status=r["status"], macs=r["macs"], parameters=r.get("parameters"), validation_macro_f1=r.get("validation", {}).get("macro_f1"), validation_accuracy=r.get("validation", {}).get("accuracy"), validation_ce=r.get("validation", {}).get("ce"), best_epoch=r.get("best_epoch"), proxy_status=r["proxy"]["status"], tree=Architecture.from_dict(r["architecture"]).tree.shape, error=r.get("error")))
            epoch_rows.extend(dict(identity, hash=r["hash"], **e) for e in r.get("history", []))
        diagnostic_groups = {"all": records}
        good = [r for r in records if r["status"] == "complete"]
        if good:
            median = float(np.median([r["macs"] for r in good]))
            diagnostic_groups.update({"cost_below_equal_median": [r for r in good if r["macs"] <= median], "cost_above_median": [r for r in good if r["macs"] > median]})
            for shape in sorted({Architecture.from_dict(r["architecture"]).tree.shape for r in good}):
                diagnostic_groups["tree:" + shape] = [r for r in good if Architecture.from_dict(r["architecture"]).tree.shape == shape]
        for group, group_records in diagnostic_groups.items():
            diagnostics.extend(dict(identity, diagnostic_group=group, interpretation="observational; not controlled ablation", **r) for r in proxy_diagnostics(group_records, cfg["report"]["top_fraction"]))
        for h in sorted(set(roles.values())):
            arch_directory = path / "refit" / h
            calibration_path = arch_directory / "calibration.json"
            if not calibration_path.exists():
                continue
            calibration = read_json(calibration_path)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            for seed in cfg["seeds"]["refit"]:
                result_path = arch_directory / f"seed_{seed}.json"
                if not result_path.exists():
                    continue
                r = read_json(result_path)
                axes[0].plot([e["epoch"] for e in r["history"]], [e["train_ce"] for e in r["history"]], label=str(seed))
                for role in r["aliases"]:
                    rows.append(dict(identity, role=role, refit_seed=seed, hash=h, evaluation_fingerprint=evaluation_fp, **{k: r["test"][k] for k in ("macro_f1", "accuracy", "balanced_accuracy", "ce")}, parameters=r["parameters"], macs=next(c["macs"] for c in records if c["hash"] == h), calibration_epochs=r["epochs"], calibration_seconds=calibration["training_seconds"], training_seconds=r["training_seconds"], training_peak_allocated_gpu_bytes=r.get("training_peak_allocated_gpu_bytes"), **r["profile"]))
                write_csv(directory / f"test_per_class_{h[:12]}_{seed}.csv", r["test"]["per_class"])
                write_csv(directory / f"confusion_{h[:12]}_{seed}.csv", [{"true_class": i, **{f"predicted_{j}": v for j, v in enumerate(c)}} for i, c in enumerate(r["test"]["confusion_matrix"])])
                if seed == cfg["seeds"]["refit"][0]:
                    matrix = np.asarray(r["test"]["confusion_matrix"])
                    axes[1].imshow(matrix, cmap="Blues"); axes[1].set(xlabel="Predicted class", ylabel="True class", title=f"Confusion (seed {seed})")
            axes[0].set(xlabel="Refit epoch", ylabel="TRAIN CE")
            if axes[0].lines:
                axes[0].legend(fontsize=7)
            _figure(fig, directory / ("refit_" + h[:12]), cfg["report"]["dpi"], cfg["backend"] == "test")
            write_csv(directory / ("calibration_" + h[:12] + ".csv"), calibration["history"])
    if len(modes) > 1 or any(len(p) > 1 for p in protocols_by_dataset.values()):
        raise ValueError("Report aggregation refuses mixed backend/data/evaluation protocols; export each protocol to a separate directory")
    # Duplicate custom runs for the same search seed are not additional independent
    # searches. Refuse pooling rather than quietly inflating n.
    identities = [(r["dataset"], r["variant"], r["role"], r["search_seed"], r["refit_seed"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate run-ids for the same search/refit seed; report one explicitly selected root")
    within, summary = nested_summary(rows)
    _comparison_figures(plot_runs, summary, output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("search_summary.csv", search_rows), ("candidate_metrics.csv", candidate_rows), ("candidate_epochs.csv", epoch_rows), ("refit_metrics.csv", rows), ("within_search.csv", within), ("results.csv", summary), ("proxy_diagnostics.csv", diagnostics)):
        write_csv(output / filename, values)
    for dataset in sorted({r["dataset"] for r in summary}):
        write_csv(output / f"{dataset}_results.csv", [r for r in summary if r["dataset"] == dataset])
    if external:
        write_csv(output / "external_baselines.csv", external_baselines(external, valid_protocols))
    atomic_json(output / "report_manifest.json", {"run_count": len(search_rows), "refit_fit_count": len({(r["dataset"], r["variant"], r["search_seed"], r["run_id"], r["hash"], r["refit_seed"]) for r in rows}), "modes": [list(x) for x in sorted(modes)], "synthetic_warning": "TEST MIXER / SYNTHETIC: plumbing verification only; no Mamba performance conclusions" if any(mode[1] for mode in modes) else None, "sources": [str(p) for p in manifests], "evaluation_fingerprints": {k: sorted(v) for k, v in valid_protocols.items()}, "selection": "validation only; TEST not used for architecture/epoch/proxy selection", "statistics": "search seed is the independent repeat; refit seeds nested within each search; repeated roles are aliases", "proxy": "both rank directions reported; no formula or architecture selected from these diagnostics", "hv": "fixed bounds across variants per dataset, reference from config; AUC includes observed evaluation time only", "igd": "not emitted: no predeclared joint empirical reference frontier"})
    return output


def _comparison_figures(runs, summary, output):
    import matplotlib.pyplot as plt
    for dataset in sorted({identity["dataset"] for identity, *_ in runs}):
        selected = [r for r in runs if r[0]["dataset"] == dataset]
        config = selected[0][3]
        synthetic = config["backend"] == "test"
        dpi = config["report"]["dpi"]
        groups = sorted({r[0]["variant"] for r in selected})
        colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(groups)}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig_p, ax_p = plt.subplots(figsize=(7, 4))
        for group in groups:
            values = [r for r in selected if r[0]["variant"] == group]
            histories = [r[1]["history"] for r in values if r[1]["history"]]
            if histories:
                common = min(map(len, histories))
                hv = np.array([[point["hv"] for point in h[:common]] for h in histories])
                x = np.arange(1, common + 1)
                mean = hv.mean(0)
                axes[0].step(x, mean, where="post", color=colors[group], label=f"{group} (n={len(histories)})")
                if len(histories) > 1:
                    std = hv.std(0, ddof=1)
                    axes[0].fill_between(x, mean - std, mean + std, step="post", color=colors[group], alpha=.15)
                for i, history in enumerate(histories):
                    axes[1].step([0] + [p["elapsed_seconds"] for p in history], [0] + [p["hv"] for p in history], where="post", color=colors[group], alpha=.6, label=group if i == 0 else None)
            for i, (_, _, records, _) in enumerate(values):
                front = sorted(pareto(records), key=lambda r: r["macs"])
                if front:
                    ax_p.plot([r["macs"] for r in front], [r["validation"]["macro_f1"] for r in front], "o-", color=colors[group], alpha=.65, label=group if i == 0 else None)
        axes[0].set(xlabel="Unique candidates (common observed range)", ylabel="HV mean +/- between-search std")
        axes[1].set(xlabel="Search seconds (individual runs)", ylabel="HV")
        for ax in axes:
            if ax.lines:
                ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        _figure(fig, output / dataset / "comparison_hv", dpi, synthetic)
        ax_p.set(xscale="log", xlabel="Analytic MACs", ylabel="Validation Macro-F1", title="Separate per-search Pareto fronts")
        if ax_p.lines:
            ax_p.legend(fontsize=7)
        ax_p.grid(alpha=.2)
        _figure(fig_p, output / dataset / "comparison_pareto", dpi, synthetic)
        selected_summary = [r for r in summary if r["dataset"] == dataset]
        if selected_summary:
            fig, ax = plt.subplots(figsize=(8, 4))
            for offset, role in enumerate(("high_f1", "knee", "low_macs")):
                positions, scores, errors = [], [], []
                for i, group in enumerate(groups):
                    row = next((r for r in selected_summary if r["variant"] == group and r["role"] == role), None)
                    if row:
                        positions.append(i + (offset - 1) * .2); scores.append(row["macro_f1_mean"]); errors.append(row["macro_f1_between_std"] or 0)
                if positions:
                    ax.errorbar(positions, scores, yerr=errors, fmt="o", capsize=3, label=role)
            ax.set_xticks(range(len(groups)), groups, rotation=15, ha="right")
            ax.set(ylabel="TEST Macro-F1: nested mean +/- search std", ylim=(0, 1)); ax.legend(fontsize=8); ax.grid(alpha=.2)
            _figure(fig, output / dataset / "comparison_refit", dpi, synthetic)
