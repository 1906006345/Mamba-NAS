from dataclasses import replace
import json
import random
import pytest
from mamba_tree_nas.architecture import Architecture, Leaf, Node, Space, chain
from mamba_tree_nas.config import load_config


def op(direction="F", state=8):
    return Node("OP", leaf=Leaf(direction, state, 2, 1))


def test_roundtrip_canonical_hash():
    a, b = op(), op("B")
    def arch(tree):
        return Architecture("point", 64, "mean", tree)
    p = arch(Node("PAR", (a, b)))
    assert Architecture.from_dict(json.loads(json.dumps(p.to_dict()))).hash == p.hash
    assert p.hash == arch(Node("PAR", (b, a))).hash
    assert arch(Node("SEQ", (a, b))).hash != arch(Node("SEQ", (b, a))).hash
    assert arch(a).hash != replace(arch(a), pooling="max").hash
    d = p.to_dict(); d["format_version"] = 99
    with pytest.raises(ValueError):
        Architecture.from_dict(d)


@pytest.mark.parametrize("bad", [{"op": "eval", "children": []}, {"op": "RES", "children": []}, {"op": "OP", "leaf": {"direction": "bad", "d_state": 8, "d_conv": 2, "expand": 1}}])
def test_bad_schema(bad):
    with pytest.raises(ValueError):
        Node.from_dict(bad)


def test_three_residual_chain_is_legal():
    c = load_config()
    tree = chain([op().leaf] * 3)
    assert tree.leaves == 3 and tree.depth == 3
    a = Architecture("point", 64, "mean", tree)
    for name in ("chain_forward", "tree_forward"):
        assert Space(c["space"], c["experiments"][name]).valid(a)


@pytest.mark.parametrize("variant", ["chain_forward", "tree_forward", "chain_direction", "tree_direction", "tree_direction_random"])
def test_genetic_operators_stay_in_space(variant):
    c = load_config()
    space = Space(c["space"], c["experiments"][variant])
    rng = random.Random(42)
    samples = [space.sample(rng) for _ in range(100)]
    for a, b in zip(samples, reversed(samples)):
        assert space.valid(a)
        assert space.valid(space.crossover(a, b, rng))
        assert space.valid(space.mutate(a, rng, c["search"]["mutation_weights"]))


def test_all_bounded_shapes_have_sampling_support():
    c = load_config()
    space = Space(c["space"], c["experiments"]["tree_direction"])
    rng = random.Random(25)
    shapes = {space.sample(rng).tree.shape for _ in range(3000)}
    assert {"OP", "RES(OP)", "SEQ(OP,OP)", "PAR(OP,OP)", "RES(RES(RES(OP)))"}.issubset(shapes)
    assert not space.valid(Architecture("point", 64, "mean", Node("RES", (chain([op().leaf] * 3),))))
