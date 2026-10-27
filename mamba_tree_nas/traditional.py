"""Conventional categorical-vector baselines on a fixed residual serial space.

The tree is only a model/serialization adapter here: sampling, crossover and
mutation operate on categorical genes, never on functional subtrees. Evaluation,
objectives, normalization, refit and artifact conventions come from tree NAS.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import time

import torch

from .architecture import Architecture, Leaf, Space, chain
from .artifacts import Events, atomic_json, error_record, init_run, read_json, run_lock, write_csv
from .config import DEFAULT_CONFIG, digest, merge, protocol_fingerprint
from .cost import COST_CONVENTION, cost_upper_bound, macs
from .metrics import hypervolume, normalized_objective, nsga_select, pareto, representatives, topology_stats, tournament
from .proxy import measure_proxy
from .search import load_records, run_path, tuple_tree
from .training import train


def baseline_config(cfg, config_path=None):
    """Extend only baseline runs; ordinary tree configs/fingerprints stay intact."""
    import yaml
    raw = yaml.safe_load(Path(config_path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    options = raw.get("traditional_baselines")
    if not options:
        raise ValueError("Configuration needs a traditional_baselines section")
    source = cfg["experiments"].get(options["serial_experiment"])
    if not source or source["topology"] != "chain":
        raise ValueError("traditional_baselines.serial_experiment must name a chain experiment")
    if not source["directions"] or not set(source["directions"]).issubset({"F", "B", "FB_shared"}):
        raise ValueError("Traditional serial directions must be chosen from F/B/FB_shared")
    variants = options["variants"]
    if set(variants) != {"nsga2", "random"} or len(set(variants.values())) != 2:
        raise ValueError("Traditional baselines require distinct nsga2/random variant names")
    if any(v in cfg["experiments"] for v in variants.values()):
        raise ValueError("Traditional baseline variant names must not replace existing experiments")
    for key in ("gene_exchange_probability", "mutation_probability", "gene_mutation_probability"):
        value = options[key]
        if value is None and key == "gene_mutation_probability":
            continue
        if not isinstance(value, (float, int)) or not 0 <= value <= 1:
            raise ValueError(f"Invalid traditional baseline probability: {key}")
    # Baseline source changes invalidate baseline resume, without changing the
    # source fingerprint of already-running functional-tree experiments.
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiments = {name: dict(source, algorithm=algorithm) for algorithm, name in variants.items()}
    return merge(cfg, {"experiments": experiments, "traditional": {**options, "source_sha256": implementation}})


class SerialVectorSpace:
    """Four global genes followed by four independent genes per residual block.

    Inactive block slots have no identity and cannot inflate the unique budget.
    If a depth change activates an absent slot, its genes are sampled uniformly.
    """
    def __init__(self, config, experiment, options):
        self.c, self.e, self.options = config, experiment, options
        self.adapter = Space(config, experiment)
        self.names = ["tokenizer", "d_model", "pooling", "num_blocks"]
        self.choices = [list(config[k]) for k in self.names[:3]] + [list(range(1, config["max_leaves"] + 1))]
        for block in range(config["max_leaves"]):
            self.names.extend(f"block_{block}.{k}" for k in ("direction", "d_state", "d_conv", "expand"))
            self.choices.extend([list(experiment["directions"])] + [list(config[k]) for k in ("d_state", "d_conv", "expand")])

    def encode(self, arch):
        if not self.adapter.valid(arch):
            raise ValueError("Architecture is outside the fixed serial space")
        from .architecture import at, paths
        leaves = [at(arch.tree, p).leaf for p in paths(arch.tree) if at(arch.tree, p).op == "OP"]
        values = [arch.tokenizer, arch.d_model, arch.pooling, len(leaves)]
        values.extend(value for leaf in leaves for value in (leaf.direction, leaf.d_state, leaf.d_conv, leaf.expand))
        return tuple(self.choices[i].index(value) for i, value in enumerate(values)) + (None,) * (len(self.names) - len(values))

    def decode(self, vector, rng=None):
        if len(vector) != len(self.names):
            raise ValueError("Wrong serial vector length")
        def get(i):
            gene = vector[i]
            if gene is None:
                if rng is None:
                    raise ValueError("New active block genes require a search RNG")
                return rng.choice(self.choices[i])
            if not isinstance(gene, int) or not 0 <= gene < len(self.choices[i]):
                raise ValueError("Serial gene index out of bounds")
            return self.choices[i][gene]
        count = get(3)
        leaves = [Leaf(*(get(4 + block * 4 + j) for j in range(4))) for block in range(count)]
        arch = Architecture(*(get(i) for i in range(3)), chain(leaves))
        if not self.adapter.valid(arch):
            raise ValueError("Decoded architecture violates serial bounds")
        return arch

    def sample(self, rng):
        count = rng.choice(self.choices[3])
        vector = [rng.randrange(len(choices)) for choices in self.choices[:4 + 4 * count]]
        vector[3] = count - 1
        return self.decode(vector + [None] * (len(self.names) - len(vector)))

    def crossover(self, first, second, rng):
        a, b = self.encode(first), self.encode(second)
        vector = [y if rng.random() < self.options["gene_exchange_probability"] else x for x, y in zip(a, b)]
        return self.decode(vector, rng)

    def mutate(self, arch, rng):
        vector = list(self.encode(arch))
        active = 4 + 4 * arch.tree.leaves
        probability = self.options["gene_mutation_probability"]
        probability = 1 / active if probability is None else probability
        for i in range(active):
            alternatives = [j for j in range(len(self.choices[i])) if j != vector[i]]
            if alternatives and rng.random() < probability:
                vector[i] = rng.choice(alternatives)
        return self.decode(vector, rng)


def new_pending(space, cfg, rng, records, population, count):
    used = {r["hash"] for r in records}
    pending = []
    for _ in range(space.c["max_attempts"]):
        if len(pending) == count:
            break
        if not population or space.e["algorithm"] == "random":
            arch = space.sample(rng)
        else:
            a = Architecture.from_dict(tournament(population, rng)["architecture"])
            b = Architecture.from_dict(tournament(population, rng)["architecture"])
            arch = space.crossover(a, b, rng) if rng.random() < cfg["search"]["crossover_probability"] else a
            if rng.random() < space.options["mutation_probability"]:
                arch = space.mutate(arch, rng)
        if arch.hash not in used:
            used.add(arch.hash)
            pending.append(arch.to_dict())
    if len(pending) != count:
        raise RuntimeError("Unique serial candidates exhausted max_attempts; reduce budget or enlarge space")
    return pending


def search(cfg, dataset, variant, seed, run_id="main", resume=False, synthetic=False, interrupt_after=None):
    origin = time.perf_counter()
    if variant not in cfg.get("traditional", {}).get("variants", {}).values():
        raise ValueError("Traditional search requires an explicitly configured traditional baseline variant")
    if cfg["backend"] == "test" and not synthetic:
        raise ValueError("Test backend is permitted only for explicitly synthetic smoke/tests")
    from .data import DataProtocol
    data = DataProtocol(cfg, dataset, synthetic)
    path = run_path(cfg, dataset, variant, seed, run_id)
    identity = {"dataset": dataset, "variant": variant, "search_seed": seed, "run_id": run_id, "synthetic": synthetic}
    with run_lock(path / ".search.lock"):
        init_run(path, cfg, identity, data.manifest, resume)
        return _search_locked(path, cfg, data, variant, seed, origin, interrupt_after)


def _search_locked(path, cfg, data, variant, seed, origin, interrupt_after):
    # This orchestration follows the same candidate commit/restart contract as
    # tree search. All numerical evaluation/selection functions are shared.
    space = SerialVectorSpace(cfg["space"], cfg["experiments"][variant], cfg["traditional"])
    upper = cost_upper_bound(cfg["space"], data.max_length, data.channels, len(data.classes))
    cache_protocol = digest({"protocol": protocol_fingerprint(cfg), "data": data.fingerprint, "traditional": cfg["traditional"]})
    rng = random.Random(seed)
    state_path = path / "search_state.json"
    state = read_json(state_path) if state_path.exists() else {"status": "running", "generation": 0, "completed": [], "population": [], "pending": [], "current_candidate": None, "elapsed_seconds": 0.0, "consecutive_failures": 0, "history": [], "rng": rng.getstate(), "cost_bound": upper, "cache_protocol": cache_protocol}
    if state["cache_protocol"] != cache_protocol or state["cost_bound"] != upper:
        raise ValueError("Traditional search cache protocol mismatch")
    if state["status"] == "complete" and (path / "representatives.json").exists():
        return path
    rng.setstate(tuple_tree(state["rng"]))
    records = load_records(path) if state_path.exists() else []
    indexed = {r["hash"]: r for r in records}
    events = Events(path / "tensorboard" / "search", cfg, len(records))
    previous = state["elapsed_seconds"]
    state["status"] = "running"
    processed = 0

    def save():
        state["elapsed_seconds"] = previous + time.perf_counter() - origin
        state["rng"] = rng.getstate()
        atomic_json(state_path, state)

    try:
        while len(records) < cfg["search"]["budget"] or state["pending"]:
            if not state["pending"]:
                population = [indexed[h] for h in state["population"]]
                count = min(cfg["search"]["population"], cfg["search"]["budget"] - len(records))
                state["pending"] = new_pending(space, cfg, rng, records, population, count)
                save()
            arch = Architecture.from_dict(state["pending"][0])
            h = arch.hash
            state["current_candidate"] = h
            save()
            candidate_path = path / "candidates" / f"{h}.json"
            cache_key = digest({"architecture": h, "evaluation": cache_protocol})
            if candidate_path.exists():
                record = read_json(candidate_path)
                if record["cache_key"] != cache_key:
                    raise ValueError("Candidate cache key mismatch")
            else:
                record = {"hash": h, "architecture": arch.to_dict(), "encoding": "categorical_serial", "gene_names": space.names, "genotype": space.encode(arch), "cache_key": cache_key, "generation": state["generation"], "evaluation_index": len(records) + 1, "macs": macs(arch, data.max_length, data.channels, len(data.classes)), "cost_convention": COST_CONVENTION, "candidate_seed": cfg["seeds"]["candidate"]}
                begin = time.perf_counter()
                record["proxy"] = measure_proxy(arch, data, cfg)
                candidate_events = Events(path / "tensorboard" / "candidates" / h[:12], cfg, 0)
                try:
                    def log_epoch(row):
                        for key, value in row.items():
                            if key != "epoch":
                                candidate_events.scalar(key, value, row["epoch"])
                        state["current_epoch"] = row["epoch"]
                        save()
                    trained = train(arch, data, cfg, cfg["seeds"]["candidate"], epoch_callback=log_epoch)
                    record.update({k: trained[k] for k in ("validation", "best_epoch", "epochs_run", "training_seconds", "training_peak_allocated_gpu_bytes", "parameters", "history")})
                    record.update(status="complete", objectives=[1 - trained["validation"]["macro_f1"], math.log1p(record["macs"])], analytic_flops=2 * record["macs"])
                    del trained
                except Exception as exc:
                    record.update(status="failed", error=error_record(exc, "candidate_training"))
                    if str(cfg["device"]).startswith("cuda"):
                        torch.cuda.empty_cache()
                finally:
                    candidate_events.close()
                record["evaluation_seconds"] = time.perf_counter() - begin
                atomic_json(candidate_path, record)
            records.append(record)
            indexed[h] = record
            state["completed"].append(h)
            state["pending"].pop(0)
            state["current_candidate"] = state["current_epoch"] = None
            state["consecutive_failures"] = state["consecutive_failures"] + 1 if record["status"] == "failed" else 0
            if not state["pending"]:
                pool = [indexed[x] for x in state["population"]] + [r for r in records if r["generation"] == state["generation"] and r["status"] == "complete"]
                pool = list({r["hash"]: r for r in pool}.values())
                state["population"] = [r["hash"] for r in nsga_select(pool, min(len(pool), cfg["search"]["population"]))]
                state["generation"] += 1
            front = pareto(records)
            hv = hypervolume([normalized_objective(r, upper) for r in front], cfg["search"]["hv_reference"])
            point = {"candidate_count": len(records), "successful_count": sum(r["status"] == "complete" for r in records), "elapsed_seconds": previous + time.perf_counter() - origin, "hv": hv, "pareto_size": len(front), "failure_rate": sum(r["status"] == "failed" for r in records) / len(records), **topology_stats(records)}
            state["history"].append(point)
            for key in ("hv", "pareto_size", "elapsed_seconds", "failure_rate", "topology_count", "topology_entropy"):
                events.scalar("search/" + key, point[key], len(records))
            for scope, values in record["proxy"].get("aggregate", {}).items():
                for key, value in values.items():
                    events.scalar(f"proxy/{scope}/{key}", value, len(records))
            events.scalar("proxy/mean_ce", record["proxy"].get("mean_ce"), len(records))
            save()
            print(f"[{data.name}/{variant}] {len(records)}/{cfg['search']['budget']} {h[:10]} {record['status']} HV={hv:.5f}", flush=True)
            processed += 1
            if state["consecutive_failures"] >= cfg["search"]["consecutive_failure_limit"]:
                raise RuntimeError("Consecutive candidate/backend failures reached limit; inspect candidate errors before resuming")
            if interrupt_after is not None and processed >= interrupt_after:
                raise InterruptedError("Injected test interruption after a committed candidate")
        chosen = representatives(records, upper)
        atomic_json(path / "representatives.json", {"roles": chosen, "architectures": {h: indexed[h]["architecture"] for h in set(chosen.values())}, "rule": "all-time true Pareto; high F1 / fixed-bound normalized distance to ideal / low MACs; deterministic MACs+hash tie break", "cost_upper_bound": upper, "hv_reference": cfg["search"]["hv_reference"]})
        write_csv(path / "search_history.csv", state["history"])
        state.update(status="complete", representatives=chosen)
        save()
        return path
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed"
        state["last_error"] = error_record(exc, "search")
        save()
        raise
    finally:
        events.close()
