from __future__ import annotations

import platform
import torch
from torch import nn
from torch.nn import functional as F
from .architecture import Architecture, Node


def prefix_mask(mask):
    if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(dim=1).all():
        raise ValueError("Expected nonempty boolean [B,L] mask")
    if (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("Only valid-prefix masks are supported")


def reverse_valid(x, mask):
    length = mask.sum(1)
    positions = torch.arange(x.shape[1], device=x.device)[None, :].expand(mask.shape[0], -1)
    indices = torch.where(mask, length[:, None] - 1 - positions, positions)
    return x.gather(1, indices[..., None].expand_as(x)) * mask[..., None]


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


class TestMixer(nn.Module):
    """CPU-only causal differentiable test double; never a Mamba performance model."""
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * expand)
        self.conv = nn.Conv1d(d_model * expand, d_model * expand, d_conv, groups=d_model * expand)
        self.out_proj = nn.Linear(d_model * expand, d_model)
        self.kernel = d_conv

    def forward(self, x):
        h = self.in_proj(x).transpose(1, 2)
        h = self.conv(F.pad(h, (self.kernel - 1, 0))).transpose(1, 2)
        return self.out_proj(F.silu(h))


def mixer_factory(backend, device):
    if backend == "test":
        if str(device) != "cpu":
            raise RuntimeError("Test mixer is CPU-only")
        return TestMixer
    if backend != "mamba" or platform.system() != "Linux" or not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Production requires Linux/WSL2, NVIDIA CUDA and official mamba-ssm; use smoke for CPU plumbing tests")
    from mamba_ssm import Mamba
    return Mamba


class Tokenizer(nn.Module):
    def __init__(self, channels, width, kind):
        super().__init__()
        self.patch = 1 if kind == "point" else int(kind.removeprefix("patch"))
        self.projection = nn.Linear(channels, width) if self.patch == 1 else nn.Conv1d(channels, width, self.patch, stride=self.patch)

    def forward(self, x, mask):
        x = x * mask[..., None]
        if self.patch == 1:
            out, valid = self.projection(x), mask
        else:
            padding = (-x.shape[1]) % self.patch
            x = F.pad(x, (0, 0, 0, padding))
            valid = F.pad(mask, (0, padding)).reshape(mask.shape[0], -1, self.patch).any(-1)
            out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return out * valid[..., None], valid


class TreeModule(nn.Module):
    def __init__(self, node: Node, width, dropout, factory):
        super().__init__()
        self.op = node.op
        self.branches = nn.ModuleList([TreeModule(c, width, dropout, factory) for c in node.children])
        if node.op == "OP":
            leaf = node.leaf
            self.direction = leaf.direction
            self.norm = RMSNorm(width)
            # Preserve the official mixer's custom parameter initialization.
            self.mixer = factory(d_model=width, d_state=leaf.d_state, d_conv=leaf.d_conv, expand=leaf.expand)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        if self.op == "OP":
            h = self.norm(x) * mask[..., None]
            if self.direction == "F":
                y = self.mixer(h)
            elif self.direction == "B":
                y = reverse_valid(self.mixer(reverse_valid(h, mask)), mask)
            else:
                y = (self.mixer(h) + reverse_valid(self.mixer(reverse_valid(h, mask)), mask)) / 2
            y = self.dropout(y)
        elif self.op == "RES":
            y = x + self.branches[0](x, mask)
        elif self.op == "SEQ":
            y = self.branches[1](self.branches[0](x, mask), mask)
        else:
            y = (self.branches[0](x, mask) + self.branches[1](x, mask)) / 2
        return y * mask[..., None]


class Classifier(nn.Module):
    def __init__(self, arch: Architecture, channels, classes, dropout=0.1, backend="mamba", device="cuda", factory=None):
        super().__init__()
        self.arch = arch
        self.tokenizer = Tokenizer(channels, arch.d_model, arch.tokenizer)
        self.tree = TreeModule(arch.tree, arch.d_model, dropout, factory or mixer_factory(backend, device))
        self.head = nn.Linear(arch.d_model * (2 if arch.pooling == "meanmax" else 1), classes)

    def forward(self, x, mask):
        prefix_mask(mask)
        x, mask = self.tokenizer(x, mask)
        x = self.tree(x, mask)
        mean = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = x.masked_fill(~mask[..., None], -torch.inf).amax(1)
        pooled = mean if self.arch.pooling == "mean" else maximum if self.arch.pooling == "max" else torch.cat((mean, maximum), -1)
        return self.head(pooled)
