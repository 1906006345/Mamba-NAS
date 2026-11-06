from dataclasses import replace
import hashlib
import json
import random

import pytest
import torch
from torch import nn

from mamba_tree_nas.architecture import Architecture, Leaf, Node, Space
from mamba_tree_nas.artifacts import atomic_json, read_json
from mamba_tree_nas.config import digest, load_config
from mamba_tree_nas.cost import cost_upper_bound, leaf_macs, macs
from mamba_tree_nas.model import TreeModule, downsample_valid
from mamba_tree_nas.multiscale import BASELINES, comparison_row


def op(rate, direction="F"):
    return Node("OP", leaf=Leaf(direction, 2, 2, 1, rate))


def test_schema_search_and_cost():
    c = load_config(profile="smoke", multiscale=True)
    tree = Node("PAR", (op(2), op(4, "FB_shared")))
    arch = Architecture("point", 8, "mean", tree, 2)
    legacy = Architecture("point", 8, "mean", op(1))
    legacy_json = {"format_version": 1, "tokenizer": "point", "d_model": 8, "pooling": "mean",
                   "tree": {"op": "OP", "leaf": {"direction": "F", "d_state": 2, "d_conv": 2, "expand": 1}}}
    assert legacy.to_dict() == legacy_json and legacy.hash == digest(legacy_json)
    assert Architecture.from_dict(arch.to_dict()).hash == arch.hash
    assert replace(arch, tree=Node("PAR", tree.children[::-1])).hash == arch.hash
    assert replace(arch, tree=Node("PAR", (op(1), tree.children[1]))).hash != arch.hash
    assert macs(arch, 33, 3, 2) == 33 * (3 + 1) * 8 + 16 + sum(
        leaf_macs(8, n, node.leaf) for n, node in zip((17, 9), tree.children))
    assert macs(arch, 33, 3, 2) <= cost_upper_bound(c["space"], 33, 3, 2)
    assert set(load_config()["experiments"]).isdisjoint(BASELINES)
    rng = random.Random(5)
    for experiment in c["experiments"].values():
        space = Space(c["space"], experiment, min_train_length=32)
        assert space.rates("point") == (1, 2, 4)
        assert space.rates("patch8") == (1,)
        for _ in range(80):
            a, b = space.sample(rng), space.sample(rng)
            child = space.crossover(a, b, rng)
            assert space.valid(child)
            for mode in ("global", "leaf", "subtree"):
                assert space.valid(space.mutate(child, rng, {mode: 1}))


class Cumulative(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x.cumsum(1) * self.gain


@pytest.mark.parametrize("rate", [1, 2, 4])
def test_mask_and_directions(rate):
    x = torch.tensor([[[1., 0.], [3., 0.], [0., 4.], [0., 6.], [8., 8.], [99., 99.]]])
    mask = torch.arange(6)[None, :] < 5
    coarse, valid = downsample_valid(x, mask, rate)
    assert valid.sum() == (5 + rate - 1) // rate
    torch.testing.assert_close(coarse[0, valid[0]][-1], x[0, 4])
    modules = [TreeModule(op(rate, d), 2, 0, Cumulative) for d in ("F", "B", "FB_shared")]
    for m in modules:
        torch.testing.assert_close(m(x, mask)[:, :5], m(x[:, :5], mask[:, :5]))
    f, b, fb = (m(x, mask) for m in modules)
    h = modules[0].norm(coarse[:, valid[0]])
    expected = [h.cumsum(1), h.flip(1).cumsum(1).flip(1)]
    for actual, target in zip((f, b), expected):
        torch.testing.assert_close(actual[:, :5], target.repeat_interleave(rate, 1)[:, :5])
    torch.testing.assert_close(fb, (f+b)/2)
    assert len({sum(p.numel() for p in m.parameters()) for m in modules}) == 1
    for mode in ("RES", "SEQ", "PAR"):
        node = Node(mode, (op(rate),) if mode == "RES" else (op(rate), op(1)))
        module = TreeModule(node, 2, 0, Cumulative)
        full = TreeModule(op(1), 2, 0, Cumulative)
        target = (x + f) * mask[..., None] if mode == "RES" else full(f, mask) if mode == "SEQ" else (f + full(x, mask))/2
        torch.testing.assert_close(module(x, mask), target)


def test_e2e(tmp_path, capsys):
    from mamba_tree_nas.cli import main
    from mamba_tree_nas.search import run_path, search

    overrides = [f"output={tmp_path / 'runs'}", f"exports={tmp_path / 'exports'}",
                 "logging.tensorboard=false", "search.population=2", "search.budget=2",
                 "seeds.refit=[31001]"]
    args = ["run-multiscale", "--profile", "smoke", "--search-seed", "2021", "--dataset", "S"]
    for value in overrides:
        args += ["--set", value]
    main(args + ["--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert {r["variant"] for r in plan["runs"]} == set(BASELINES)
    assert not (tmp_path / "runs").exists()
    cfg = load_config(profile="smoke", overrides=overrides, multiscale=True)
    dataset = "S"
    variant = "tree_multiscale_forward"
    with pytest.raises(InterruptedError):
        search(cfg, dataset, variant, 2021, synthetic=True, interrupt_after=1)
    main(args)
    output = tmp_path / "exports" / "multiscale" / "main"
    assert (output / "report_manifest.json").is_file()
    new = run_path(cfg, dataset, variant, 2021, "main")
    assert read_json(new / "refit/state.json")["status"] == "complete"
    assert read_json(new / "search_state.json")["history"][-1]["scale_structure_counts"]
    old = run_path(cfg, dataset, BASELINES[variant], 2021, "main")
    manifest = read_json(new / "manifest.json")
    manifest["identity"]["variant"] = BASELINES[variant]
    atomic_json(old / "manifest.json", manifest)
    state = read_json(new / "search_state.json")
    atomic_json(old / "search_state.json", state)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir()}
    main(args + ["--report-only"])
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir()}
    assert comparison_row(new, old)["hv_delta"] == 0
    state["cost_bound"] += 1
    atomic_json(old / "search_state.json", state)
    assert "hv_delta" not in comparison_row(new, old)
