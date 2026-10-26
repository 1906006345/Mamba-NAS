from dataclasses import replace
import pytest
import torch
from torch import nn
from mamba_tree_nas.architecture import Architecture, Leaf, Node
from mamba_tree_nas.model import Classifier, TreeModule, reverse_valid, prefix_mask


class CumSum(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x.cumsum(1) * self.scale


def leaf(direction="F"):
    return Node("OP", leaf=Leaf(direction, 2, 2, 1))


def test_composition_semantics():
    x = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    def build(node):
        result = TreeModule(node, 4, 0, CumSum)
        # Remove normalization to test only composition semantics.
        for m in result.modules():
            if isinstance(m, TreeModule) and m.op == "OP":
                m.norm = nn.Identity()
        return result
    f = build(leaf())
    assert torch.allclose(f(x, mask), x.cumsum(1))
    assert torch.allclose(build(Node("RES", (leaf(),)))(x, mask), x + f(x, mask))
    assert torch.allclose(build(Node("SEQ", (leaf(), leaf())))(x, mask), f(f(x, mask), mask))
    assert torch.allclose(build(Node("PAR", (leaf(), leaf())))(x, mask), f(x, mask))


def test_direction_and_shared_parameters():
    x = torch.tensor([[[1.], [2.], [3.], [0.], [0.]]])
    mask = torch.tensor([[True, True, True, False, False]])
    def model(d):
        m = TreeModule(leaf(d), 1, 0, CumSum); m.norm = nn.Identity(); return m
    f, b, fb = model("F"), model("B"), model("FB_shared")
    assert torch.equal(b(x, mask), torch.tensor([[[6.], [5.], [3.], [0.], [0.]]]))
    assert torch.equal(fb(x, mask), (f(x, mask) + b(x, mask)) / 2)
    assert sum(p.numel() for p in f.parameters()) == sum(p.numel() for p in fb.parameters())
    assert torch.equal(reverse_valid(reverse_valid(x, mask), mask), x)


@pytest.mark.parametrize("tokenizer", ["point", "patch8", "patch16"])
@pytest.mark.parametrize("direction", ["F", "B", "FB_shared"])
def test_padding_invariance_and_backward(tokenizer, direction):
    arch = Architecture(tokenizer, 8, "meanmax", Node("RES", (leaf(direction),)))
    model = Classifier(arch, 3, 2, 0, "test", "cpu").eval()
    x = torch.randn(1, 17, 3)
    mask = torch.ones(1, 17, dtype=torch.bool)
    extra = torch.cat((x, torch.randn(1, 15, 3) * 99), 1)
    extra_mask = torch.cat((mask, torch.zeros(1, 15, dtype=torch.bool)), 1)
    y = model(x, mask)
    assert torch.allclose(y, model(extra, extra_mask), atol=1e-6)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mask_rejection():
    with pytest.raises(ValueError):
        prefix_mask(torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError):
        prefix_mask(torch.zeros(1, 3, dtype=torch.bool))


def test_different_leaves_do_not_share_parameters():
    tree = TreeModule(Node("PAR", (leaf("FB_shared"), leaf("FB_shared"))), 4, 0, CumSum)
    assert tree.branches[0].mixer.scale is not tree.branches[1].mixer.scale
