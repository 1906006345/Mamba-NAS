import math
from .architecture import at, paths


def leaf_macs(width, tokens, leaf):
    inner = width * leaf.expand
    rank = math.ceil(width / 16)
    per_token = 2 * width * inner + leaf.d_conv * inner + inner * (rank + 2 * leaf.d_state) + rank * inner + 3 * inner * leaf.d_state + inner + inner * width
    return tokens * per_token * (2 if leaf.direction == "FB_shared" else 1)


def macs(arch, length, channels, classes):
    patch = 1 if arch.tokenizer == "point" else int(arch.tokenizer[5:])
    tokens = math.ceil(length / patch)
    width = arch.d_model
    trunk = sum(leaf_macs(width, tokens, at(arch.tree, p).leaf) for p in paths(arch.tree) if at(arch.tree, p).op == "OP")
    pool = tokens * width * (2 if arch.pooling == "meanmax" else 1)
    return int(tokens * patch * channels * width + trunk + pool + width * (2 if arch.pooling == "meanmax" else 1) * classes)


def cost_upper_bound(space, length, channels, classes):
    from .architecture import Architecture, Leaf, Node
    # A valid conservative bound for every comparison group, including all PAR
    # branches and shared reverse scans. Covers the shortest patch's ceil effect.
    width = max(space["d_model"])
    leaf = Leaf("FB_shared", max(space["d_state"]), max(space["d_conv"]), max(space["expand"]))
    return max(macs(Architecture(t, width, "meanmax", Node("OP", leaf=leaf)), length, channels, classes) + (space["max_leaves"] - 1) * leaf_macs(width, math.ceil(length / (1 if t == "point" else int(t[5:]))), leaf) for t in space["tokenizer"])


COST_CONVENTION = "Analytic MAC estimate at maximum TRAIN length; FLOPs=2*MACs. Includes projection, causal conv, selective scan approximation, gating, pooling and classifier; excludes norms, bias, residual/merge arithmetic and kernel overhead. Not a complete fused-kernel profiler count."
