import torch
from torch import nn
from mamba_ssm import Mamba
from einops.layers.torch import Rearrange
from einops import rearrange


class PredictionHead(nn.Module):
    def __init__(self, in_nf, out_nf):
        super().__init__()
        self.linear = nn.Linear(in_nf, out_nf)

    def forward(self, x, n_vars):
        x = rearrange(x, 'b (c l) d -> (b c) (l d)', c=n_vars)
        x = self.linear(x)
        x = rearrange(x, '(b c) p -> b p c', c=n_vars)

        return x


class Model(nn.Module):
    """
    Paper link: http://arxiv.org/abs/2405.16440
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # ----- 计算实际的 patch 数量（考虑 stride）-----
        self.num_patches = (configs.seq_len - configs.patch_len) // configs.stride + 1

        # patching and embedding
        self.value_embedding = nn.Sequential(
            Rearrange('b c l d -> (b c) l d'),
            nn.Linear(configs.patch_len, configs.d_model, bias=False),
        )

        # Encoder. The source repository provides MixerTSModel privately; the
        # unified runner uses the same official Mamba blocks as other models.
        self.encoder = nn.ModuleList([
            Mamba(
                d_model=configs.d_model,
                d_state=configs.d_state,
                d_conv=configs.d_conv,
                expand=configs.expand,
            )
            for _ in range(configs.e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(configs.d_model)
        self.classifier = (
            nn.Linear(configs.d_model, configs.num_class)
            if self.task_name == 'classification' else None
        )

        # Prediction Head – 使用正确的 patch 数量计算输入维度
        self.head = PredictionHead(configs.d_model * self.num_patches, configs.pred_len)


    def forecast(self, x_enc, x_mark_enc=None):
        b, _, n_vars = x_enc.shape

        # Norm
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # Do patching and embedding
        x = x_enc.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        enc_in = self.value_embedding(x)

        # Variable Scan along Time (VST)
        enc_in = rearrange(enc_in, '(b c) l d -> b (c l) d', c=n_vars)

        # Encoder
        for layer in self.encoder:
            enc_in = layer(enc_in)
        enc_out = self.encoder_norm(enc_in)

        # Decoder
        dec_out = self.head(enc_out, n_vars=n_vars)

        # De-norm
        dec_out = dec_out * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, [0], :].repeat(1, self.pred_len, 1))

        return dec_out

    def classification(self, x_enc, padding_mask=None):
        if x_enc.ndim != 3:
            raise ValueError('Expected x_enc with shape [B, L, C]')
        if padding_mask is not None:
            x_enc = x_enc * padding_mask.unsqueeze(-1).to(x_enc.dtype)
        x = x_enc.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        batch_size, n_vars, num_patches, _ = x.shape
        enc_in = self.value_embedding(x)
        enc_in = rearrange(enc_in, '(b c) l d -> b (c l) d', b=batch_size, c=n_vars)
        for layer in self.encoder:
            enc_in = layer(enc_in)
        enc_in = self.encoder_norm(enc_in)
        enc_in = rearrange(enc_in, 'b (c l) d -> b c l d', c=n_vars, l=num_patches)
        pooled = enc_in.mean(dim=(1, 2))
        return self.classifier(pooled)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return self.forecast(x_enc, x_mark_enc)

    def batch_update_state(self, cost_tensor):
        for layer in self.encoder:
            if hasattr(layer, 'batch_update_state'):
                layer.batch_update_state(cost_tensor)

    def set_reordering_index(self, reordering_index):
        for layer in self.encoder:
            if hasattr(layer, 'set_reordering_index'):
                layer.set_reordering_index(reordering_index)

    def reset_ids_shuffle(self):
        for layer in self.encoder:
            if hasattr(layer, 'reset_ids_shuffle'):
                layer.reset_ids_shuffle()
