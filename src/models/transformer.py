import torch
from torch_geometric.nn import MLP
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, ModuleList, Identity
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)

from src.models.components.selfmha import MultiHeadSelfAttention
from src.models.components.ffn import FeedForwardBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers=1,
        skip_connections=False,
        activation="relu",
        num_heads=8,
        use_ffn=True,
        ff_hidden_dim=256,
        dropout=0.0,
        norm="batch_norm",
        bias=True,
        init_embed_hidden_channels=[512, 256],
    ) -> None:
        super().__init__()
        if norm == "None":
            norm = None
        self.mlp = MLP(
            [-1, *init_embed_hidden_channels, latent_dim],
            norm=None,
            plain_last=True,
            act=activation,
        )
        self.layers = ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    latent_dim,
                    activation=activation,
                    num_heads=num_heads,
                    use_ffn=use_ffn,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    norm=norm,
                    bias=bias,
                )
            )
        self.skip_connections = skip_connections
        self.attn_masks = [None, None, None]

    def forward(self, x, edge_index=None, batch=None, id=None, *args, **kwargs):
        x = self.mlp(x)
        # make sure the shape of x is correct
        # convert node features to dense tensor
        # TODO: mask attention weights for sim graph
        for layer in self.layers:
            if self.skip_connections:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        activation="relu",
        num_heads=8,
        use_ffn=True,
        ff_hidden_dim=256,
        dropout=0.0,
        norm="batch_norm",
        bias=True,
    ) -> None:
        super().__init__()
        self.selfmha = MultiHeadSelfAttention(
            latent_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            bias=bias,
        )
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = FeedForwardBlock(
                latent_dim, ff_hidden_dim, activation=activation, dropout=dropout
            )
        if norm is None:
            self.norm1 = Identity()
            self.norm2 = Identity()
        else:
            self.norm1 = normalization_resolver(norm, in_channels=latent_dim)
            self.norm2 = normalization_resolver(norm, in_channels=latent_dim)

    def forward(self, x, **kwargs):
        out, _ = self.selfmha(x)
        if self.use_ffn:
            out = self.norm1(x + out)
            out_ffn = self.ffn(out)
            out = self.norm2(out + out_ffn)
        return out
