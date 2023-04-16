from torch_geometric.nn import MLP
import torch.nn as nn


class CelltypeDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_celltypes,
        num_layers=2,
        hidden_channels=64,
        activation="relu",
        norm=None,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_celltypes = num_celltypes
        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_celltypes,
            num_layers=num_layers,
            norm=norm,
            plain_last=True,
            act=activation,
            **kwargs
        )

    def forward(self, x):
        logits = self.mlp(x)

        return logits


class GeneExpressionDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_genes,
        activation="relu",
        norm=None,
        dropout=0.0,
        hidden_channels=[256, 512],
        **kwargs
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.mlp = MLP(
            channel_list=[latent_dim, *hidden_channels, num_genes],
            norm=norm,
            plain_last=True,
            act=activation,
            dropout=dropout,
            **kwargs
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
