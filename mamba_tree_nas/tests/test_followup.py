from dataclasses import replace
import random

from mamba_tree_nas.architecture import Architecture, Leaf, Node, Space
from mamba_tree_nas.config import load_config
from mamba_tree_nas.followup import leaves, ordered_dict, with_rates


def test_fixed_rate_swap_changes_only_rate():
    a = Architecture("patch8", 64, "meanmax", Node("PAR", (
        Node("OP", leaf=Leaf("F", 8, 2, 1, 2)),
        Node("RES", (Node("OP", leaf=Leaf("B", 16, 4, 2, 4)),)),
    )), 2)
    b = with_rates(a, (4, 2))
    assert a.tree.shape == b.tree.shape
    assert [leaf.rate for _, leaf in leaves(b)] == [4, 2]
    assert with_rates(b, (2, 4)) == a
    assert Architecture.from_dict(ordered_dict(b)) == b
    assert [replace(leaf, rate=1) for _, leaf in leaves(a)] == [
        replace(leaf, rate=1) for _, leaf in leaves(b)]


def test_single_rate_does_not_advance_extra_rng():
    c = load_config(multiscale=True)
    e = c["experiments"]["tree_multiscale_direction"]
    a, b = random.Random(11), random.Random(11)
    plain = Space(c["space"], {k: v for k, v in e.items() if k != "multiscale"})
    fixed = Space(c["space"], {**e, "multiscale": {"min_tokens": 8, "rates": [1]}}, 144)
    assert plain.leaf(a) == fixed.leaf(b, "point")
    assert a.getstate() == b.getstate()
    assert Space(c["space"], e, 144).rates("patch8") == (1, 2)


def test_fixed_refit_exports_and_reuses(cfg, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mamba_tree_nas import followup
    from mamba_tree_nas.data import DataProtocol

    c = load_config(profile="smoke", multiscale=True)
    c["logging"]["tensorboard"] = False
    c["seeds"]["refit"] = [31001]
    variant = "tree_multiscale_forward"
    identity = {"dataset": "S", "synthetic": True, "variant": variant}
    manifest = {"config": c, "identity": identity, "data": DataProtocol(c, "S", True).manifest}
    arch = Architecture("point", 8, "mean", Node("OP", leaf=Leaf("F", 2, 2, 1, 4)), 2)
    monkeypatch.setattr(followup, "fixed_plan", lambda *_: [("case", manifest, arch, [(4,), (1,)])])
    args = SimpleNamespace(source_root=tmp_path / "old", source_run_id="main", execute=False)
    output = tmp_path / "f"
    followup.fixed(args, output)
    assert not output.exists()
    args.execute = True
    followup.fixed(args, output)
    before = (output / "fixed_paired.csv").read_bytes()
    followup.fixed(args, output)
    assert before == (output / "fixed_paired.csv").read_bytes()
    assert not args.source_root.exists()
