import torch
from torch_geometric.nn import GCNConv, GCN, MLP, TransformerConv, GATv2Conv
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, ModuleList, Identity
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch.nn import MultiheadAttention
from torch_geometric.nn import knn_graph


class DissectSpatial(nn.Module):
    def __init__(
        self,
        num_celltypes,
        latent_dim,
        num_encoder_layers=2,
        num_decoder_layers=1,
        activation="relu",
        encoder_kwargs={},
        decoder_kwargs={},
        use_pos=True,
        fusion="concat",
        reset_parameters=False,
    ) -> None:
        super().__init__()
        self.encoder = BiChannelGNNEncoder(
            latent_dim,
            num_encoder_layers,
            activation=activation,
            use_pos=use_pos,
            fusion=fusion,
            **encoder_kwargs
        )
        self.decoder = CelltypeDecoder(
            latent_dim,
            int(latent_dim / 2),
            num_celltypes,
            activation=activation,
            num_layers=num_decoder_layers,
            **decoder_kwargs
        )
        self.use_pos = use_pos
        # optional custom initialization
        if reset_parameters:
            self.reset_parameters()

    def forward(
        self, x, edge_index, edge_weight=None, edge_attr=None, pos=None, batch=None
    ):
        if pos is not None and self.use_pos:
            x = torch.cat([x, pos], dim=-1)
        z = self.encoder(x, edge_index, edge_weight, edge_attr, pos, batch)
        out = self.decoder(z)
        return out

    def reset_parameters(self):
        for w in self.parameters():
            if len(w.shape) == 1:
                zeros(w)
            else:
                glorot(w)


class BiChannelGNNEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        ff_hidden_dim=256,
        # batch_size,
        activation="relu",
        dropout=0.0,
        norm="batch_norm",
        spatial_conv_kwargs={},
        latent_conv_kwargs={},
        use_pos=True,
        fusion="concat",
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            [-1, 512, 256, latent_dim],
            norm=None,
            plain_last=True,
            act=activation,
        )
        self.layers = ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                BiChannelGNNBlock(
                    latent_dim,
                    ff_hidden_dim,
                    # batch_size=batch_size,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    spatial_conv_kwargs=spatial_conv_kwargs,
                    latent_conv_kwargs=latent_conv_kwargs,
                    use_pos=use_pos,
                    fusion=fusion,
                )
            )

    def forward(
        self, x, edge_index, edge_weight=None, edge_attr=None, pos=None, batch=None
    ):
        x = self.mlp(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight, edge_attr, pos, batch)
            # maybe add skip connection
        return x


class BiChannelGNNBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        ff_hidden_dim,
        activation="relu6",
        dropout=0.0,
        norm="layer",
        num_heads=1,
        spatial_conv_kwargs={},
        latent_conv_kwargs={},
        knn=6,
        cosine=False,
        use_pos=True,
        fusion="concat",
        sim_layer=False,
    ) -> None:
        super().__init__()
        self.spatial_conv = GATv2Conv(
            latent_dim,
            latent_dim,
            num_heads,
            edge_dim=1,
            add_self_loops=False,
            **spatial_conv_kwargs
        )
        # extra conv with just self loops for simulated data
        if sim_layer:
            self.spatial_lin = nn.Linear(latent_dim, latent_dim)
        self.sim_layer = sim_layer
        self.activation = activation_resolver(activation)
        
        # self.latent_conv = GATv2Conv(
        #     latent_dim, latent_dim, num_heads, edge_dim=1, **latent_conv_kwargs
        # )
        # without latent graph construction
        # self.latent_conv = MultiheadAttention(
        #     latent_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True,
        #     **latent_conv_kwargs
        # )
        

        if fusion == "concat":
            self.concat_linear = nn.LazyLinear(latent_dim)
        elif fusion == "gating":
            self.gate = GatingUnit(num_inputs=2)
        self.ffn = FeedForwardBlock(
            latent_dim, ff_hidden_dim, activation=activation, dropout=dropout
        )

        if norm is None:
            self.norm1 = Identity()
            self.norm2 = Identity()
        else:
            self.norm1 = normalization_resolver(norm, in_channels=latent_dim)
            self.norm2 = normalization_resolver(norm, in_channels=latent_dim)
        self.knn = knn
        self.cosine = cosine
        self.use_pos = use_pos
        self.fusion = fusion

    def construct_latent_graph(self, x, batch):
        # compute pairwise distances between all embeddings
        # use k-nearest neighbors to construct graph
        edge_index = knn_graph(
            x,
            k=self.knn,
            batch=batch,
            loop=False,
            flow="source_to_target",
            cosine=self.cosine,
        )
        return edge_index

    def forward(
        self, x, edge_index, edge_weight=None, edge_attr=None, pos=None, batch=None
    ):
        # x.shape = (batch*num_nodes, latent_dim)

        # perform spatial and latent GNN computation
        # latent_edge_index = self.construct_latent_graph(x, batch)
        # out_latent = self.latent_conv(x, latent_edge_index)

        # TODO: check if conv takes edge weights or edge attributes
        if self.sim_layer and edge_attr is None:
            out_spatial = self.spatial_lin(x)
        else:
            out_spatial = self.spatial_conv(x, edge_index, edge_attr=edge_attr)

        if self.fusion == "gating":
            out = self.gate(x, out_spatial)
        elif self.fusion == "concat":
            # concat fusion
            # concat gnn outputs, perform linear transformation, and add initial input
            out_concat = torch.cat([out_spatial], dim=-1)
            out_multi = self.concat_linear(out_concat)
            out = x + out_multi
        else:
            out = out_spatial
            out = self.activation(out)

        # perform normalization and feed forward
        out = self.norm1(out)
        out_ffn = self.ffn(out)
        out = self.norm2(out + out_ffn)

        return out


class GatingUnit(nn.Module):
    def __init__(self, num_inputs=2) -> None:
        super().__init__()
        self.linear = nn.LazyLinear(num_inputs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, *args):
        gate = self.linear(x)
        gate = self.softmax(gate)
        # compute weighted sum
        out = gate[:, [0]] * x
        for i in range(1, len(args)):
            out += gate[:, [i]] * args[i]
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0, activation="relu") -> None:
        super().__init__()
        self.linear1 = Linear(d_model, hidden_dim)
        self.linear2 = Linear(hidden_dim, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation_resolver(activation)

    def forward(self, x: Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class CelltypeDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_celltypes,
        num_layers=1,
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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.softmax(x)

        return x
