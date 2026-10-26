from __future__ import annotations

from dataclasses import dataclass, replace
import random
from .config import digest

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Leaf:
    direction: str
    d_state: int
    d_conv: int
    expand: int

    def __post_init__(self):
        if self.direction not in ("F", "B", "FB_shared") or min(self.d_state, self.d_conv, self.expand) < 1:
            raise ValueError("Invalid leaf configuration")


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple["Node", ...] = ()
    leaf: Leaf | None = None

    def __post_init__(self):
        arity = {"OP": 0, "RES": 1, "SEQ": 2, "PAR": 2}
        if self.op not in arity or len(self.children) != arity[self.op] or (self.leaf is not None) != (self.op == "OP"):
            raise ValueError("Malformed tree node")

    def to_dict(self):
        if self.op == "OP":
            return {"op": "OP", "leaf": vars(self.leaf)}
        children = [c.to_dict() for c in self.children]
        if self.op == "PAR":
            children.sort(key=digest)
        return {"op": self.op, "children": children}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise ValueError("Tree must be an object")
        if d.get("op") == "OP":
            if set(d) != {"op", "leaf"}:
                raise ValueError("Unknown OP fields")
            return cls("OP", leaf=Leaf(**d["leaf"]))
        if set(d) != {"op", "children"}:
            raise ValueError("Unknown composition fields")
        return cls(d["op"], tuple(cls.from_dict(x) for x in d["children"]))

    @property
    def leaves(self):
        return 1 if self.op == "OP" else sum(c.leaves for c in self.children)

    @property
    def depth(self):
        return 0 if self.op == "OP" else 1 + max(c.depth for c in self.children)

    @property
    def shape(self):
        if self.op == "OP":
            return "OP"
        values = [c.shape for c in self.children]
        if self.op == "PAR":
            values.sort()
        return f"{self.op}({','.join(values)})"


@dataclass(frozen=True)
class Architecture:
    tokenizer: str
    d_model: int
    pooling: str
    tree: Node

    def __post_init__(self):
        if self.tokenizer not in ("point", "patch8", "patch16") or self.pooling not in ("mean", "max", "meanmax") or self.d_model < 1:
            raise ValueError("Invalid global configuration")

    def to_dict(self):
        return {"format_version": FORMAT_VERSION, "tokenizer": self.tokenizer, "d_model": self.d_model, "pooling": self.pooling, "tree": self.tree.to_dict()}

    @classmethod
    def from_dict(cls, d):
        if set(d) != {"format_version", "tokenizer", "d_model", "pooling", "tree"} or d["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported architecture schema")
        return cls(d["tokenizer"], d["d_model"], d["pooling"], Node.from_dict(d["tree"]))

    @property
    def hash(self):
        return digest(self.to_dict())


def paths(tree, prefix=()):
    yield prefix
    for i, c in enumerate(tree.children):
        yield from paths(c, prefix + (i,))


def at(tree, path):
    for index in path:
        tree = tree.children[index]
    return tree


def substitute(tree, path, value):
    if not path:
        return value
    children = list(tree.children)
    children[path[0]] = substitute(children[path[0]], path[1:], value)
    return replace(tree, children=tuple(children))


def chain(leaves):
    nodes = [Node("RES", (Node("OP", leaf=x),)) for x in leaves]
    result = nodes[-1]
    for node in reversed(nodes[:-1]):
        result = Node("SEQ", (node, result))
    return result


def is_chain(node):
    if node.op == "RES" and node.children[0].op == "OP":
        return True
    return node.op == "SEQ" and is_chain(node.children[0]) and is_chain(node.children[1])


def canonical_chain(node):
    if not is_chain(node):
        return node
    def leaves(n):
        return [n.children[0].leaf] if n.op == "RES" else leaves(n.children[0]) + leaves(n.children[1])
    return chain(leaves(node))


class Space:
    def __init__(self, config, experiment):
        self.c = config
        self.e = experiment

    def valid(self, a):
        c = self.c
        if a.tree.leaves > c["max_leaves"] or a.tree.depth > c["max_depth"]:
            return False
        if self.e["topology"] == "chain" and not is_chain(a.tree):
            return False
        if any(getattr(a, key) not in c[key] for key in ("tokenizer", "d_model", "pooling")):
            return False
        return all(n.leaf.direction in self.e["directions"] and all(getattr(n.leaf, key) in c[key] for key in ("d_state", "d_conv", "expand")) for n in (at(a.tree, p) for p in paths(a.tree)) if n.op == "OP")

    def leaf(self, rng):
        return Leaf(rng.choice(self.e["directions"]), *(rng.choice(self.c[k]) for k in ("d_state", "d_conv", "expand")))

    def node(self, rng, depth=0):
        # Preserve the declared recursive grammar distribution, conditioning on
        # the bounds by rejecting the whole sample, rather than forcing OP.
        op = rng.choices(list(self.c["sampling"]), weights=list(self.c["sampling"].values()))[0]
        if depth >= self.c["max_depth"] and op != "OP":
            raise ValueError("Sample exceeds composition-depth bound")
        if op == "OP":
            return Node(op, leaf=self.leaf(rng))
        return Node(op, tuple(self.node(rng, depth + 1) for _ in range(1 if op == "RES" else 2)))

    def sample(self, rng):
        for _ in range(self.c["max_attempts"]):
            try:
                tree = chain([self.leaf(rng) for _ in range(rng.randint(1, self.c["max_leaves"]))]) if self.e["topology"] == "chain" else self.node(rng)
            except ValueError:
                continue
            a = Architecture(*(rng.choice(self.c[k]) for k in ("tokenizer", "d_model", "pooling")), tree)
            if self.valid(a):
                return a
        raise RuntimeError("Cannot sample a legal tree; check bounds/distribution")

    def crossover(self, a, b, rng):
        for _ in range(64):
            tree = substitute(a.tree, rng.choice(list(paths(a.tree))), at(b.tree, rng.choice(list(paths(b.tree)))))
            child = Architecture(*(rng.choice((getattr(a, k), getattr(b, k))) for k in ("tokenizer", "d_model", "pooling")), tree)
            if self.e["topology"] == "chain":
                child = replace(child, tree=canonical_chain(child.tree))
            if self.valid(child):
                return child
        return a

    def mutate(self, a, rng, weights):
        mode = rng.choices(list(weights), weights=list(weights.values()))[0]
        for _ in range(64):
            if mode == "global":
                key = rng.choice(("tokenizer", "d_model", "pooling"))
                child = replace(a, **{key: rng.choice(self.c[key])})
            elif mode == "leaf":
                path = rng.choice([p for p in paths(a.tree) if at(a.tree, p).op == "OP"])
                child = replace(a, tree=substitute(a.tree, path, Node("OP", leaf=self.leaf(rng))))
            elif self.e["topology"] == "chain":
                child = replace(a, tree=chain([self.leaf(rng) for _ in range(rng.randint(1, self.c["max_leaves"]))]))
            else:
                path = rng.choice(list(paths(a.tree)))
                try:
                    child = replace(a, tree=substitute(a.tree, path, self.node(rng, len(path))))
                except ValueError:
                    continue
            if self.valid(child):
                return child
        return a
