import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, ModuleList, Identity
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.nn import knn_graph
from torch_geometric.nn.dense.linear import Linear as PygLinear
from torch_geometric.nn import GCNConv, GCN, MLP, TransformerConv, GATv2Conv

from src.models.components.fusion import GatingUnit
from src.models.components.ffn import FeedForwardBlock
from src.models.components.selfmha import MultiHeadSelfAttention


class MultiChannelGNNEncoder(nn.Module):
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
        mha_channel=False,
        latent_channel=False,
        knn=6,
        cosine=False,
        latent_conv_kwargs={},
        fusion=None,
        use_ffn=False,
        ff_hidden_dim=256,
        plain_last=False,
        norm_first=False,
        init_embed_hidden_channels=[512, 256],
        inter_skip=False,
        use_pos=False,
        sim_pos_enc=False,
        use_id=False,
        use_sparse=False,
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
                MultiChannelGNNBlock(
                    latent_dim,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    lin_channel=lin_channel,
                    spatial_channel=spatial_channel,
                    num_heads=num_heads,
                    spatial_conv_kwargs=spatial_conv_kwargs,
                    mha_channel=mha_channel,
                    latent_channel=latent_channel,
                    knn=knn,
                    cosine=cosine,
                    latent_conv_kwargs=latent_conv_kwargs,
                    fusion=fusion,
                    use_ffn=use_ffn,
                    ff_hidden_dim=ff_hidden_dim,
                    plain_last=plain_last,
                    norm_first=norm_first,
                    use_sparse=use_sparse,
                )
            )
        self.use_pos = use_pos
        if use_pos:
            self.pos_encoder = nn.Linear(2, latent_dim)
            if sim_pos_enc:
                self.sim_pos_enc = nn.Linear(2, latent_dim)
            else:
                self.sim_pos_enc = None
            # self.sim_pos_token = nn.Parameter(torch.FloatTensor(1, latent_dim))
            # torch.nn.init.xavier_uniform_(self.sim_pos_token)
        self.use_id = use_id
        if use_id:
            self.id_encoder = nn.Linear(3, latent_dim)
        self.inter_skip = inter_skip

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
        # x.shape = (batch*num_nodes, latent_dim)
        # id.shape = (batch*num_nodes, 3)
        # pos.shape = (batch*num_nodes, 2)

        x = self.encode_latent(x)
        if self.use_pos:
            if pos[0, 0] == -1 and self.sim_pos_enc is not None:
                x_pos = self.sim_pos_enc(pos)
            else:
                x_pos = self.pos_encoder(pos)
            x_spatial = x + x_pos
        else:
            x_spatial = None
        

        for k, layer in enumerate(self.layers):
            # only add id and pos encoding to first layer input
            if k == 0:
                if self.use_id:
                    x = x + self.id_encoder(id)
            else:
                x_spatial = None
            out_layer = layer(
                x,
                edge_index,
                edge_weight,
                edge_attr,
                pos,
                batch,
                x_spatial=x_spatial,
            )
            if self.inter_skip:
                x = x + out_layer
            else:
                x = out_layer
        return x

    def encode_latent(self, x):
        return self.mlp(x)


class MultiChannelGNNBlock(nn.Module):
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
        mha_channel=False,
        latent_channel=False,
        knn=6,
        cosine=False,
        latent_conv_kwargs={},
        fusion=None,
        use_ffn=False,
        ff_hidden_dim=256,
        plain_last=False,
        norm_first=False,
        use_sparse=False,
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
            self.lin_layer = PygLinear(
                latent_dim, latent_dim, weight_initializer="glorot"
            )

        self.latent_channel = latent_channel
        if latent_channel:
            self.latent_conv = GATv2Conv(
                latent_dim, latent_dim, num_heads, edge_dim=1, **latent_conv_kwargs
            )

        self.mha_channel = mha_channel
        if mha_channel:
            self.mha = MultiHeadSelfAttention(
                latent_dim, num_heads, dropout=dropout, add_bias_kv=True
            )

        # make sure some channel is selected
        num_channels = sum([spatial_channel, lin_channel, latent_channel, mha_channel])
        assert num_channels > 0, "No channels selected"

        self.fusion = fusion if fusion is not None else ""
        if fusion == "gating":
            self.gating_unit = GatingUnit()
        else:
            self.concat_linear = nn.LazyLinear(latent_dim)

        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = FeedForwardBlock(
                latent_dim, ff_hidden_dim, activation=activation, dropout=dropout
            )

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
        self.norm_first = norm_first
        self.use_sparse = use_sparse

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        edge_attr=None,
        pos=None,
        batch=None,
        x_spatial=None,
    ):
        # x.shape = (batch*num_nodes, latent_dim)
        if self.norm_first:
            x = self.norm1(x)

        # TODO: check if conv takes edge weights or edge attributes
        channels = []
        if self.spatial_channel:
            if x_spatial is None:
                x_spatial = x
            if self.use_sparse:
                adj = SparseTensor(
                    row=edge_index[0], col=edge_index[1], value=edge_attr
                )
                out_spatial = self.spatial_conv(x_spatial, adj.t())
            else:
                out_spatial = self.spatial_conv(
                    x_spatial, edge_index, edge_attr=edge_attr
                )
            channels.append(out_spatial)
            # might need projection back to latent dim
        if self.lin_channel:
            channels.append(self.lin_layer(x))
        if self.latent_channel:
            latent_edge_index = self.construct_latent_graph(x, batch)
            # might need projection back to latent dim
            channels.append(self.latent_conv(x, latent_edge_index))
        if self.mha_channel:
            channels.append(self.mha(x)[0])

        # fusion should be dependent on the type of the input: real, sim, or mix
        # maybe input extra onehot encoding of type of input

        if self.fusion == "gating":
            out_fusion = self.gating_unit(x, *channels)
        elif self.fusion == "concat":
            out_concat = torch.cat([x, *channels], dim=-1)
            out_fusion = self.concat_linear(out_concat)
        elif self.fusion == "concat_skip":
            out_concat = torch.cat(channels, dim=-1)
            out_fusion = self.concat_linear(out_concat)
            out_fusion = out_fusion + x
        elif self.fusion == "concat_simple":
            out_concat = torch.cat(channels, dim=-1)
            out_fusion = self.concat_linear(out_concat)
        else:
            out_fusion = self.concat_linear(channels[0])

        # perform normalization and feed forward
        if self.use_ffn:
            if self.norm_first:
                out = self.norm2(out_fusion)
                out = out + self.ffn(out)
            else:
                out = self.norm1(out_fusion)
                out = self.norm2(out + self.ffn(out))
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
