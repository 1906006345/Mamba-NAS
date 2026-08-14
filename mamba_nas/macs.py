from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .model import token_length
from .search_space import Genome


@dataclass(frozen=True)
class MacBreakdown:
    tokenizer: int
    mamba: int
    pooling: int
    classifier: int

    @property
    def total(self) -> int:
        return self.tokenizer + self.mamba + self.pooling + self.classifier

    def to_dict(self) -> dict:
        return {**asdict(self), "total": self.total}


def estimate_macs(
    genome: Genome, sequence_length: int, input_channels: int, num_classes: int
) -> MacBreakdown:
    """Consistent analytical MAC proxy used for every candidate.

    The selective scan term counts state update and readout operations. Elementwise
    activations/norms are intentionally excluded and this convention is exported.
    """
    tokens = token_length(sequence_length, genome.tokenizer)
    patch = 1 if genome.tokenizer == "point" else int(genome.tokenizer.removeprefix("patch"))
    tokenizer = tokens * genome.d_model * input_channels * patch

    d_inner = genome.expand * genome.d_model
    dt_rank = math.ceil(genome.d_model / 16)
    per_token = (
        2 * genome.d_model * d_inner  # in_proj creates x and gate
        + genome.d_conv * d_inner  # depthwise temporal conv
        + d_inner * (dt_rank + 2 * genome.d_state)  # x_proj
        + dt_rank * d_inner  # dt_proj
        + 3 * d_inner * genome.d_state  # state update, input, readout
        + d_inner  # gate
        + d_inner * genome.d_model  # out_proj
    )
    scans = 2 if genome.direction == "bidirectional" else 1
    mamba = tokens * per_token * genome.num_blocks * scans
    pooling = tokens * genome.d_model * (2 if genome.pooling == "meanmax" else 1)
    head_dimension = genome.d_model * (2 if genome.pooling == "meanmax" else 1)
    classifier = head_dimension * num_classes
    return MacBreakdown(int(tokenizer), int(mamba), int(pooling), int(classifier))


def trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
