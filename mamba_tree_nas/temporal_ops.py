"""Local time-resolution edits for multiscale functional trees."""
from dataclasses import replace
from itertools import product

from .architecture import Leaf, at, paths, substitute
from .cost import macs, trunk_macs


def leaf_paths(tree, prefix=()):
    return [p for p in paths(tree, prefix) if at(tree, p[len(prefix):]).op == "OP"]


def _edits(space, arch, mode):
    tree = arch.tree
    leaves = leaf_paths(tree)
    rates = sorted(space.rates(arch.tokenizer))

    def steps(p):
        i = rates.index(at(tree, p).leaf.rate)
        return [(d, rates[i+d]) for d in (-1, 1) if 0 <= i+d < len(rates)]

    if mode == "legacy_leaf":
        for p, values in product(leaves, product(space.e["directions"], space.c["d_state"],
                                                  space.c["d_conv"], space.c["expand"], rates)):
            yield [(p, Leaf(*values))]
    elif mode == "single":
        for p in leaves:
            for _, r in steps(p):
                yield [(p, replace(at(tree, p).leaf, rate=r))]
    elif mode in ("random_double", "redistribute"):
        for p in paths(tree):
            if at(tree, p).op != "PAR":
                continue
            for left, right in product(*(leaf_paths(at(tree, p + (i,)), p + (i,)) for i in (0, 1))):
                for (d1, r1), (d2, r2) in product(steps(left), steps(right)):
                    if mode == "random_double" or d1 != d2:
                        yield [(left, replace(at(tree, left).leaf, rate=r1)),
                               (right, replace(at(tree, right).leaf, rate=r2))]
    else:
        raise ValueError(f"Unknown temporal mutation: {mode}")


def neighbors(space, arch, mode, dimensions, window, used=()):
    """Return distinct feasible architectures and counts before/after filtering."""
    cost = ((lambda a: trunk_macs(a, dimensions[0]))
            if mode in ("random_double", "redistribute") else (lambda a: macs(a, *dimensions)))
    base = cost(arch)
    found, seen = {}, {arch.hash}
    counts = dict(proposed=0, invalid=0, equivalent=0, outside_window=0, already_seen=0)
    for edits in _edits(space, arch, mode):
        tree = arch.tree
        for p, leaf in edits:
            tree = substitute(tree, p, replace(at(tree, p), leaf=leaf))
        child = replace(arch, tree=tree)
        h = child.hash
        counts["proposed"] += 1
        if h in seen:
            counts["equivalent"] += 1
            continue
        seen.add(h)
        if not space.valid(child):
            counts["invalid"] += 1
        elif abs(cost(child) - base) > base * window:
            counts["outside_window"] += 1
        elif h in used:
            counts["already_seen"] += 1
        else:
            found[h] = child
    counts["feasible"] = len(found)
    return found, counts
