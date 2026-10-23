import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding

class Model(nn.Module):
    """Multi-layer Mamba model with the repository's standard interface.

    The original version projected every encoded input token and then selected
    the last ``pred_len`` tokens. Those tokens are still history positions, so
    that path did not actually decode future timestamps. The forecasting head
    below instead maps the final context state directly to all future steps.

    ``e_layers`` now controls the number of sequential Mamba blocks.  The first
    block remains registered as ``self.mamba`` so checkpoints created by the
    previous single-layer implementation remain compatible when
    ``e_layers == 1``.  Additional blocks are stored separately.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.num_class = getattr(configs, "num_class", None)

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"
        self.e_layers = int(getattr(configs, "e_layers", 1))
        if self.e_layers < 1:
            raise ValueError(f"e_layers must be at least 1, got {self.e_layers}")
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # Classification receives variable-length padded batches. A pointwise
        # projection avoids the circular temporal convolution in DataEmbedding
        # leaking suffix padding into the valid prefix.
        self.classification_embedding = nn.Linear(configs.enc_in, configs.d_model)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )
        self.extra_mamba_layers = nn.ModuleList(
            Mamba(
                d_model=configs.d_model,
                d_state=configs.d_ff,
                d_conv=configs.d_conv,
                expand=configs.expand,
            )
            for _ in range(self.e_layers - 1)
        )

        self.c_out = configs.c_out
        self.forecast_head = (
            nn.Linear(configs.d_model, self.pred_len * self.c_out, bias=False)
            if self.task_name in {"short_term_forecast", "long_term_forecast"}
            else None
        )
        if self.task_name == "classification":
            if self.num_class is None or self.num_class < 2:
                raise ValueError("classification requires configs.num_class >= 2")
            self.classifier = nn.Linear(configs.d_model, self.num_class)

    @property
    def mamba_layers(self):
        """Return all Mamba blocks in execution order without re-registering them."""
        return (self.mamba, *self.extra_mamba_layers)

    def _encode_mamba(self, values):
        for layer in self.mamba_layers:
            values = layer(values)
        return values

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self._encode_mamba(x)
        # Use the encoded context at the end of the observed window to predict
        # all horizons at once: [B, d_model] -> [B, pred_len, c_out].
        context = x[:, -1, :]
        x_out = self.forecast_head(context).view(
            x.shape[0], self.pred_len, self.c_out
        )

        # For multivariate (M) forecasting c_out == enc_in. For MS forecasting
        # c_out is commonly 1, so denormalize with the target channel statistics.
        if self.c_out == x_enc.shape[-1]:
            target_mean, target_std = mean_enc, std_enc
        elif self.c_out == 1:
            target_mean, target_std = mean_enc[..., -1:], std_enc[..., -1:]
        else:
            target_mean = mean_enc[..., : self.c_out]
            target_std = std_enc[..., : self.c_out]
        return x_out * target_std + target_mean

    def classification(self, x_enc, padding_mask):
        """Encode a padded variable-length series and return class logits."""
        if x_enc.ndim != 3:
            raise ValueError(f"Expected x_enc [B, L, C], got {tuple(x_enc.shape)}")
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
        mean_enc = (x_enc * mask3).sum(dim=1, keepdim=True).detach() / denominator.unsqueeze(-1)
        centered = (x_enc - mean_enc) * mask3
        variance = (centered.square()).sum(dim=1, keepdim=True) / denominator.unsqueeze(-1)
        std_enc = torch.sqrt(variance + 1e-5).detach()
        normalized = centered / std_enc
        encoded = self.classification_embedding(normalized)
        encoded = encoded * mask3
        encoded = self._encode_mamba(encoded)
        encoded = encoded * mask3
        pooled = encoded.sum(dim=1) / denominator
        return self.classifier(pooled)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            return self.forecast(x_enc, x_mark_enc)
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)

        raise NotImplementedError(f"Mamba model does not support task {self.task_name!r}")
