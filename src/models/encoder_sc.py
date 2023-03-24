import torch
from torch_geometric.nn import MLP
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ModuleList, Identity


class SingleCellEncoder(nn.Module):
    def __init__(
        self,
        num_genes,
        latent_dim,
        hidden_channels=[512, 256],
        activation="relu",
        norm=None,
        dropout=0.0,
        plain_last=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            channel_list=[num_genes, *hidden_channels, latent_dim],
            norm=norm,
            plain_last=plain_last,
            act=activation,
            dropout=dropout,
            **kwargs
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
