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

from src.models.components.fusion import GatingUnit
from src.models.decoder import CelltypeDecoder
from src.models.components.ffn import FeedForwardBlock
from src.models.transformer import TransformerEncoder


class DissectSpatial(nn.Module):
    def __init__(
        self,
        num_celltypes,
        latent_dim,
        activation="relu",
        use_pos=True,
        encoder_type="gnn",
        use_id=False,
        encoder_kwargs={},
        decoder_kwargs={},
    ) -> None:
        super().__init__()
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                latent_dim, activation=activation, **encoder_kwargs
            )
        elif encoder_type == "gnn":
            self.encoder = BiChannelGNNEncoder(
                latent_dim, activation=activation, **encoder_kwargs
            )
        else:
            raise ValueError(
                f"encoder_type {encoder_type} not supported, choose from ['transformer', 'gnn']"
            )
        self.decoder = CelltypeDecoder(
            latent_dim, num_celltypes, activation=activation, **decoder_kwargs
        )
        self.use_pos = use_pos
        self.use_id = use_id

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        edge_attr=None,
        pos=None,
        batch=None,
        id=None,
    ):
        if pos is not None and self.use_pos:
            x = torch.cat([x, pos], dim=-1)
        if id is not None and self.use_id:
            x = torch.cat([x, id], dim=-1)
        z = self.encoder(
            x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            pos=pos,
            batch=batch,
        )
        out = self.decoder(z)
        return out

    def reset_parameters(self):
        # requires that all parameters are initialized beforehand
        for w in self.parameters():
            if len(w.shape) == 1:
                zeros(w)
            else:
                glorot(w)


class BiChannelGNNEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers=1,
        activation="relu6",
        dropout=0.0,
        norm=None,
        lin_channel=False,
        spatial_channel=True,
        num_heads=1,
        spatial_conv_kwargs={},
        latent_channel=False,
        knn=6,
        cosine=False,
        latent_conv_kwargs={},
        fusion=None,
        use_ffn=False,
        ff_hidden_dim=256,
        plain_last=False,
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
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    lin_channel=lin_channel,
                    spatial_channel=spatial_channel,
                    num_heads=num_heads,
                    spatial_conv_kwargs=spatial_conv_kwargs,
                    latent_channel=latent_channel,
                    knn=knn,
                    cosine=cosine,
                    latent_conv_kwargs=latent_conv_kwargs,
                    fusion=fusion,
                    use_ffn=use_ffn,
                    ff_hidden_dim=ff_hidden_dim,
                    plain_last=plain_last,
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
        activation="relu6",
        dropout=0.0,
        norm=None,
        lin_channel=False,
        spatial_channel=True,
        num_heads=1,
        spatial_conv_kwargs={},
        latent_channel=False,
        knn=6,
        cosine=False,
        latent_conv_kwargs={},
        fusion=None,
        use_ffn=False,
        ff_hidden_dim=256,
        plain_last=False,
    ) -> None:
        super().__init__()
        self.spatial_channel = spatial_channel
        if spatial_channel:
            self.spatial_conv = GATv2Conv(
                latent_dim,
                latent_dim,
                num_heads,
                edge_dim=1,
                add_self_loops=False,
                **spatial_conv_kwargs,
            )

        # extra conv with just self loops for simulated data
        self.lin_channel = lin_channel
        if lin_channel:
            self.lin_layer = nn.Linear(latent_dim, latent_dim)

        self.latent_channel = latent_channel
        if latent_channel:
            self.latent_conv = GATv2Conv(
                latent_dim, latent_dim, num_heads, edge_dim=1, **latent_conv_kwargs
            )
        # make sure some channel is selected
        num_channels = sum([spatial_channel, lin_channel, latent_channel])
        assert num_channels > 0, "No channels selected"

        self.fusion = fusion
        if "concat" in fusion:
            self.concat_linear = nn.LazyLinear(latent_dim)
        elif fusion == "gating":
            self.gating_unit = GatingUnit(num_inputs=num_channels)

        if use_ffn:
            self.ffn = FeedForwardBlock(
                latent_dim, ff_hidden_dim, activation=activation, dropout=dropout
            )
        self.use_ffn = use_ffn

        self.activation = activation_resolver(activation)
        if norm is None:
            self.norm1 = Identity()
            self.norm2 = Identity()
        else:
            self.norm1 = normalization_resolver(norm, in_channels=latent_dim)
            self.norm2 = normalization_resolver(norm, in_channels=latent_dim)

        self.knn = knn
        self.cosine = cosine
        self.plain_last = plain_last

    def forward(
        self, x, edge_index, edge_weight=None, edge_attr=None, pos=None, batch=None
    ):
        # x.shape = (batch*num_nodes, latent_dim)

        # TODO: check if conv takes edge weights or edge attributes
        channels = []
        if self.spatial_channel:
            channels.append(self.spatial_conv(x, edge_index, edge_attr=edge_attr))
            # might need projection back to latent dim
        if self.lin_channel:
            channels.append(self.lin_layer(x))
        if self.latent_channel:
            latent_edge_index = self.construct_latent_graph(x, batch)
            out_latent = self.latent_conv(x, latent_edge_index)
            # might need projection back to latent dim
            channels.append(out_latent)

        # fusion should be dependent on the type of the input: real, sim, or mix
        # maybe input extra onehot encoding of type of input

        if self.fusion == "gating":
            out_fusion = self.gating_unit(x, *channels)
        elif self.fusion == "concat":
            channels.append(x)
            out_concat = torch.cat(channels, dim=-1)
            out_fusion = self.concat_linear(out_concat)
        elif self.fusion == "concat_skip":
            out_concat = torch.cat(channels, dim=-1)
            out_fusion = self.concat_linear(out_concat)
            out_fusion = out_fusion + x
        elif self.fusion == "concat_simple":
            out_concat = torch.cat(channels, dim=-1)
            out_fusion = self.concat_linear(out_concat)
        else:
            out_fusion = channels[0]

        # perform normalization and feed forward
        if self.use_ffn:
            out = self.norm1(out_fusion)
            out_ffn = self.ffn(out)
            out = self.norm2(out + out_ffn)
        else:
            out = out_fusion

        if not self.plain_last:
            out = self.activation(out)

        return out

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
