from __future__ import annotations

import math
from dataclasses import replace
from collections.abc import Callable

import torch
from torch import nn

from .search_space import Genome

MixerFactory = Callable[[Genome], nn.Module]


def official_mamba_factory(genome: Genome) -> nn.Module:
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise RuntimeError(
            "Official mamba_ssm is required for training. Install the WSL2 environment "
            "from environment-mamba-nas.yml; the repository MambaSimple is not used."
        ) from exc
    return Mamba(
        d_model=genome.d_model,
        d_state=genome.d_state,
        d_conv=genome.d_conv,
        expand=genome.expand,
    )


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def reverse_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse each valid prefix while leaving padding at the suffix."""
    lengths = mask.long().sum(dim=1)
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    reversed_positions = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
    gather = torch.where(mask, reversed_positions, positions)
    return x.gather(1, gather.unsqueeze(-1).expand_as(x)) * mask.unsqueeze(-1).to(x.dtype)


class Tokenizer(nn.Module):
    def __init__(self, input_channels: int, d_model: int, tokenizer: str, scan_order: str = "time_major"):
        super().__init__()
        self.kind = tokenizer
        self.scan_order = scan_order
        self.input_channels = input_channels
        self.patch_size = 1 if tokenizer == "point" else int(tokenizer.removeprefix("patch"))
        if scan_order == "patch_interleaved":
            if self.patch_size == 1:
                raise ValueError("patch_interleaved does not support point tokenizer")
            self.projection = nn.Linear(self.patch_size, d_model)
        elif self.patch_size == 1:
            self.projection = nn.Linear(input_channels, d_model)
        else:
            self.projection = nn.Conv1d(
                input_channels, d_model, kernel_size=self.patch_size, stride=self.patch_size
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x * mask.unsqueeze(-1).to(x.dtype)
        if self.scan_order == "patch_interleaved":
            padding = (-x.shape[1]) % self.patch_size
            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, padding))
                mask = torch.nn.functional.pad(mask, (0, padding), value=False)
            patch_mask = mask.reshape(mask.shape[0], -1, self.patch_size).any(dim=-1)
            patches = x.transpose(1, 2).unfold(-1, self.patch_size, self.patch_size)
            tokens = self.projection(patches)
            # patch0:c0,c1,..., patch1:c0,c1,...
            tokens = tokens.permute(0, 2, 1, 3).reshape(
                x.shape[0], -1, tokens.shape[-1]
            )
            token_mask = patch_mask.unsqueeze(-1).expand(
                -1, -1, self.input_channels
            ).reshape(mask.shape[0], -1)
            return tokens * token_mask.unsqueeze(-1).to(tokens.dtype), token_mask
        if self.patch_size == 1:
            return self.projection(x) * mask.unsqueeze(-1).to(x.dtype), mask
        padding = (-x.shape[1]) % self.patch_size
        if padding:
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        token_mask = mask.reshape(mask.shape[0], -1, self.patch_size).any(dim=-1)
        tokens = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return tokens * token_mask.unsqueeze(-1).to(tokens.dtype), token_mask


class ResidualMambaBlock(nn.Module):
    def __init__(self, genome: Genome, mixer_factory: MixerFactory, dropout: float):
        super().__init__()
        self.direction = genome.direction
        self.norm = RMSNorm(genome.d_model)
        self.mixer = mixer_factory(genome)
        self.dropout = nn.Dropout(dropout)

    def _scan(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x * mask.unsqueeze(-1).to(x.dtype)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        forward = self._scan(x, mask)
        if self.direction == "bidirectional":
            reversed_x = reverse_valid(x, mask)
            backward = reverse_valid(self._scan(reversed_x, mask), mask)
            mixed = 0.5 * (forward + backward)
        else:
            mixed = forward
        output = x + self.dropout(mixed)
        return output * mask.unsqueeze(-1).to(output.dtype)


class MambaTSCClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        genome: Genome,
        dropout: float = 0.1,
        mixer_factory: MixerFactory | None = None,
    ):
        super().__init__()
        self.genome = genome.validate()
        factory = mixer_factory or official_mamba_factory
        self.tokenizer = Tokenizer(
            input_channels, genome.d_model, genome.tokenizer, genome.scan_order
        )
        self.blocks = nn.ModuleList(
            ResidualMambaBlock(genome, factory, dropout) for _ in range(genome.num_blocks)
        )
        head_dimension = genome.d_model * (2 if genome.pooling == "meanmax" else 1)
        self.classifier = nn.Linear(head_dimension, num_classes)

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask3 = mask.unsqueeze(-1)
        if self.genome.pooling in ("mean", "meanmax"):
            denominator = mask3.sum(dim=1).clamp_min(1)
            mean = (x * mask3.to(x.dtype)).sum(dim=1) / denominator
        if self.genome.pooling in ("max", "meanmax"):
            maximum = x.masked_fill(~mask3, -torch.inf).amax(dim=1)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        if self.genome.pooling == "mean":
            return mean
        if self.genome.pooling == "max":
            return maximum
        return torch.cat((mean, maximum), dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or mask.shape != x.shape[:2]:
            raise ValueError(f"Expected x [B,L,C] and mask [B,L], got {x.shape}, {mask.shape}")
        x, token_mask = self.tokenizer(x, mask.bool())
        for block in self.blocks:
            x = block(x, token_mask)
        return self.classifier(self.pool(x, token_mask))


# Fixed, non-searched reference configuration.  Keeping this in the same module
# as the searched classifier guarantees that the baseline and NAS candidates use
# exactly the same masking, normalization layer, residual block and classifier
# implementation.  Only the architecture choices are frozen.
BASELINE_GENOME = Genome(
    tokenizer="point",
    num_blocks=1,
    direction="forward",
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2,
    pooling="mean",
)


class MambaBaselineClassifier(MambaTSCClassifier):
    """The fixed Mamba baseline used for comparison with Mamba-NAS.

    It deliberately has no architecture-search or candidate-dependent state.
    The optional ``mixer_factory`` is useful for CPU unit tests; production runs
    use the official ``mamba_ssm.Mamba`` implementation inherited from the base
    classifier.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        tokenizer: str = "point",
        mixer_factory: MixerFactory | None = None,
    ):
        if tokenizer not in {"point", "patch8", "patch16"}:
            raise ValueError(
                f"Invalid baseline tokenizer={tokenizer!r}; "
                "expected 'point', 'patch8', or 'patch16'"
            )
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            genome=replace(BASELINE_GENOME, tokenizer=tokenizer),
            dropout=dropout,
            mixer_factory=mixer_factory,
        )


def token_length(
    sequence_length: int,
    tokenizer: str,
    scan_order: str = "time_major",
    input_channels: int = 1,
) -> int:
    patch = 1 if tokenizer == "point" else int(tokenizer.removeprefix("patch"))
    tokens = math.ceil(sequence_length / patch)
    return tokens * input_channels if scan_order == "patch_interleaved" else tokens
