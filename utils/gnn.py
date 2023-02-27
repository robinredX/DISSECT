import torch
from torch_geometric.nn import GCNConv, GCN, MLP, TransformerConv
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


class Dissect(nn.Module):
    def __init__(
        self, num_genes, num_celltypes, channel_list=[512, 256, 128, 64], act="relu6"
    ) -> None:
        super().__init__()
        channel_list = [num_genes] + channel_list + [num_celltypes]
        self.mlp = self.mlp = MLP(
            channel_list,
            norm=None,
            plain_last=True,
            act=act,
        )
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def forward(self, x, edge_index=None, edge_weight=None):
        x = self.mlp(x[:, 0:-2])
        x = self.softmax(x)
        return x

    def reset_parameters(self):
        for w in self.parameters():
            if len(w.shape) == 1:
                zeros(w)
            else:
                glorot(w)


class DissectSpatial(nn.Module):
    def __init__(
        self, num_genes, num_celltypes, latent_dim, encoder_kwargs={}, decoder_kwargs={}
    ) -> None:
        super().__init__()
        # add 2 for the spatial coordinates
        self.encoder = GNNEncoder(num_genes + 2, latent_dim, **encoder_kwargs)
        self.decoder = CelltypeDecoder(
            latent_dim, latent_dim, num_celltypes, **decoder_kwargs
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.encoder(x, edge_index, edge_weight)
        x = self.decoder(x)
        return x


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim,
        num_layers=3,
        plain_last=True,
        parallel_channels=1,
        act="relu",
    ) -> None:
        super().__init__()
        self.spatial_convs = ModuleList()
        self.latent_convs = ModuleList()
        self.mlp = MLP(
            [in_channels, 512, 256, latent_dim], norm=None, plain_last=True, act=act
        )
        self.act = activation_resolver(act)
        self.plain_last = plain_last
        self.parallel_channels = parallel_channels

        for _ in range(num_layers):
            self.spatial_convs.append(GCNConv(latent_dim, latent_dim, improved=False))

        for _ in range(num_layers):
            self.latent_convs.append(TransformerConv(latent_dim, latent_dim))

        self.linear_final = Linear((parallel_channels + 1) * latent_dim, latent_dim)

    def spatial_channel(self, init_embed, edge_index, edge_weight=None):
        # perform spatial GNN computation
        out = self.spatial_convs[0](init_embed, edge_index, edge_weight)
        out = self.act(out)
        for conv in self.spatial_convs[1:-1]:
            out = conv(out, edge_index, edge_weight)
            out = self.act(out)
        if len(self.spatial_convs) > 1:
            out = self.spatial_convs[-1](out, edge_index, edge_weight)
        return out

    def latent_channel(self, init_embed):
        # perform latent GNN computation
        out = init_embed
        return out

    def linear_channel(self, init_embed):
        # perform linear computation
        out = init_embed
        return out

    def forward(self, x, edge_index, edge_weight=None):
        # first embed input into latent space
        init_embed = self.mlp(x)
        # init_embed.shape = (batch*num_nodes, latent_dim)

        channel_outputs = []

        # add skip connection
        channel_outputs.append(init_embed)

        if self.parallel_channels > 0:
            # perform spatial GNN computation
            out_spatial = self.spatial_channel(init_embed, edge_index, edge_weight)
            # out_spatial.shape = (batch*num_nodes, latent_dim)
            channel_outputs.append(out_spatial)

        if self.parallel_channels > 1:
            # perform latent GNN computation
            out_latent = self.latent_channel(init_embed)
            # out_latent.shape = (batch*num_nodes, latent_dim)
            channel_outputs.append(out_latent)

        # maybe add skip connection or concat inputs
        out = torch.concat(channel_outputs, dim=-1)

        # perform final linear transformation with activation
        out = self.act(out)
        out = self.linear_final(out)

        if not self.plain_last:
            out = self.act(out)

        return out


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
            norm=None,
            plain_last=True,
            **kwargs
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.softmax(x)
        return x


class GNNBlock(nn.Module):
    def __init__(
        self, in_channels, latent_dim, d_model, dim_feedforward, dropout=0.0
    ) -> None:
        super().__init__()
        self.conv = GCNConv()
        self.mlp = MLP(norm=None, dropout=dropout)
        self.activation = activation_resolver("relu")
        self.norm = normalization_resolver("batch_norm")

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight):
        # implement transformer like architecture
        x_1 = self.conv(x, edge_index, edge_weight)
        x_1 = self.norm(x + x_1)
        # x_1 = self.activation(x_1)
        x_2 = self.mlp(x_1)
        x_2 = self.norm(x_1 + x_2)
        # x_2 = self.activation(x_2)
        return x_2

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)
