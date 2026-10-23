"""Standalone fixed Mamba baseline for Mamba-NAS comparisons.

Unlike the search implementation, this model has no dependency on
``mamba_nas.model`` or its ``Genome``/NAS blocks. It follows the same
``Model(configs)`` interface as the other models in ``models/`` and receives
all Mamba hyperparameters from the normal experiment arguments.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError:  # Keep imports usable for code-only tooling and tests.
    Mamba = None


class Model(nn.Module):
    """Fixed forward Mamba classifier with configurable model capacity."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name != "classification":
            raise NotImplementedError(
                "MambaBaseline currently supports only the classification task"
            )
        if Mamba is None:
            raise RuntimeError(
                "MambaBaseline requires mamba_ssm. Install it in the CUDA environment."
            )

        self.tokenizer = self._build_tokenizer(configs)
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=configs.d_model,
                    d_state=getattr(configs, "d_state", 16),
                    d_conv=configs.d_conv,
                    expand=configs.expand,
                )
                for _ in range(max(1, configs.e_layers))
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(configs.d_model) for _ in self.layers]
        )
        self.dropout = nn.Dropout(getattr(configs, "dropout", 0.1))
        self.pooling = getattr(configs, "baseline_pooling", "mean")
        if self.pooling not in {"mean", "max", "meanmax"}:
            raise ValueError(
                f"Invalid baseline_pooling={self.pooling!r}; expected mean, max, or meanmax"
            )
        head_dim = configs.d_model * (2 if self.pooling == "meanmax" else 1)
        self.classifier = nn.Linear(head_dim, configs.num_class)

    @staticmethod
    def _build_tokenizer(configs) -> nn.Module:
        use_patch = bool(getattr(configs, "baseline_use_patch", False))
        patch_size = int(getattr(configs, "baseline_patch_size", 16))
        if not use_patch:
            return nn.Linear(configs.enc_in, configs.d_model)
        if patch_size not in (8, 16):
            raise ValueError("baseline_patch_size must be 8 or 16")
        return nn.Conv1d(
            configs.enc_in,
            configs.d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @property
    def uses_patch(self) -> bool:
        return isinstance(self.tokenizer, nn.Conv1d)

    def _tokenize(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = padding_mask.bool()
        x = x * mask.unsqueeze(-1).to(x.dtype)
        if not self.uses_patch:
            return self.tokenizer(x) * mask.unsqueeze(-1).to(x.dtype), mask

        patch_size = self.tokenizer.kernel_size[0]
        padding = (-x.shape[1]) % patch_size
        if padding:
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        token_mask = mask.reshape(mask.shape[0], -1, patch_size).any(dim=-1)
        tokens = self.tokenizer(x.transpose(1, 2)).transpose(1, 2)
        return tokens * token_mask.unsqueeze(-1).to(tokens.dtype), token_mask

    def classification(
        self, x_enc: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        if x_enc.ndim != 3:
            raise ValueError(f"Expected x_enc [B,L,C], got {tuple(x_enc.shape)}")
        if padding_mask is None:
            padding_mask = torch.ones(
                x_enc.shape[:2], device=x_enc.device, dtype=torch.bool
            )
        padding_mask = padding_mask.to(device=x_enc.device).bool()
        if padding_mask.shape != x_enc.shape[:2]:
            raise ValueError(
                f"Expected padding mask {tuple(x_enc.shape[:2])}, got {tuple(padding_mask.shape)}"
            )

        mask3 = padding_mask.unsqueeze(-1).to(x_enc.dtype)
        denominator = padding_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x_enc.dtype)
        mean = (x_enc * mask3).sum(dim=1, keepdim=True) / denominator.unsqueeze(-1)
        centered = (x_enc - mean) * mask3
        variance = centered.square().sum(dim=1, keepdim=True) / denominator.unsqueeze(-1)
        x_enc = centered / torch.sqrt(variance + 1e-5)

        x, token_mask = self._tokenize(x_enc, padding_mask)
        token_mask3 = token_mask.unsqueeze(-1).to(x.dtype)
        for norm, layer in zip(self.norms, self.layers):
            update = layer(norm(x * token_mask3))
            x = (x + self.dropout(update)) * token_mask3

        pooled_denominator = token_mask3.sum(dim=1).clamp_min(1)
        if self.pooling in {"mean", "meanmax"}:
            pooled_mean = (x * token_mask3).sum(dim=1) / pooled_denominator
        if self.pooling in {"max", "meanmax"}:
            pooled_max = x.masked_fill(~token_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
            pooled_max = torch.where(
                torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max)
            )
        if self.pooling == "mean":
            pooled = pooled_mean
        elif self.pooling == "max":
            pooled = pooled_max
        else:
            pooled = torch.cat((pooled_mean, pooled_max), dim=-1)
        return self.classifier(pooled)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        raise NotImplementedError(
            f"MambaBaseline does not support task {self.task_name!r}"
        )

