import torch
from torch_geometric.nn import GCNConv, GCN, MLP
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


class CelltypeDeconvolver(nn.Module):
    def __init__(self, num_genes, num_celltypes, latent_dim) -> None:
        super().__init__()
        # add 2 for the spatial coordinates
        self.encoder = GNNEncoder(num_genes + 2, latent_dim)
        self.decoder = CelltypeDecoder(latent_dim, latent_dim, num_celltypes)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.encoder(x, edge_index, edge_weight)
        x = self.decoder(x)
        return x


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, num_layers=2, dropout=0.0) -> None:
        super().__init__()
        # self.gnn_blocks = ModuleList()
        self.convs = ModuleList()
        self.linear = Linear(in_channels, latent_dim, bias=False)
        self.act = activation_resolver("relu")

        for _ in range(num_layers):
            self.convs.append(GCNConv(latent_dim, latent_dim, improved=False))

    def forward(self, x, edge_index, edge_weight=None):
        # first embed input into latent space
        x = self.linear(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.act(x)
        return x


class CelltypeDecoder(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, num_celltypes, num_layers=2, **kwargs
    ) -> None:
        super().__init__()
        self.num_celltypes = num_celltypes
        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_celltypes,
            num_layers=num_layers,
            **kwargs
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.softmax(x)
        return x


# class GNNBlock(nn.Module):
#     def __init__(self, in_channels, latent_dim, dropout=0.0) -> None:
#         super().__init__()
#         self.conv = GCNConv()
#         self.mlp = MLP(norm=None, dropout=dropout)
#         self.act = activation_resolver("relu")
#         self.norm = normalization_resolver("batch_norm")


#     def forward(self, x, edge_index, edge_weight):
#         # implement transformer like architecture
#         x_1 = self.conv(x, edge_index, edge_weight)
#         x_1 = self.norm(x + x_1)
#         # x_1 = self.act(x_1)
#         x_2 = self.mlp(x_1)
#         x_2 = self.norm(x_1 + x_2)
#         # x_2 = self.act(x_2)
#         return x_2
