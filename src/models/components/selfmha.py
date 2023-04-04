import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch import Tensor


class MultiHeadSelfAttention(MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask=None,
        attn_mask=None,
        average_attn_weights: bool = True,
    ):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        query = x
        key = x
        value = x
        out, attn_weights = super().forward(
            query,
            key,
            value,
            key_padding_mask,
            True,
            attn_mask,
            average_attn_weights,
        )
        return out.squeeze(0), attn_weights
