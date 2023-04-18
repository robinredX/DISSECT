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

from src.models.components.fusion import GatingUnit, FusionComponent
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
        spatial_channel_kwargs={},
        mha_channel=False,
        mha_channel_kwargs={},
        latent_channel=False,
        latent_channel_kwargs={},
        fusion=None,
        use_ffn=False,
        ff_hidden_dim=256,
        plain_last=False,
        norm_first=False,
        norm_last=False,
        init_embed_hidden_channels=[512, 256],
        inter_skip=False,
        use_pos=False,
        sim_pos_enc=False,
        use_id=False,
        use_sparse=True,
        **kwargs,
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
                    spatial_channel_kwargs=spatial_channel_kwargs,
                    mha_channel=mha_channel,
                    mha_channel_kwargs=mha_channel_kwargs,
                    latent_channel=latent_channel,
                    latent_channel_kwargs=latent_channel_kwargs,
                    fusion=fusion,
                    use_ffn=use_ffn,
                    ff_hidden_dim=ff_hidden_dim,
                    plain_last=plain_last,
                    norm_first=norm_first,
                    norm_last=norm_last,
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
        spatial_channel_kwargs={},
        mha_channel=False,
        mha_channel_kwargs={},
        latent_channel=False,
        latent_channel_kwargs={},
        fusion=None,
        use_ffn=False,
        ff_hidden_dim=256,
        plain_last=False,
        norm_first=False,
        norm_last=False,
        use_sparse=True,
        # **kwargs,
    ) -> None:
        super().__init__()

        self.spatial_channel = None
        if spatial_channel:
            self.spatial_channel = GNNChannel(
                latent_dim, heads=num_heads, **spatial_channel_kwargs
            )

        self.lin_channel = None
        if lin_channel:
            self.lin_channel = PygLinear(
                latent_dim, latent_dim, weight_initializer="glorot"
            )
        
        self.latent_channel = None
        if latent_channel:
            self.latent_channel = LatentChannel(
                latent_dim, **latent_channel_kwargs
            )

        self.mha_channel = None
        if mha_channel:
            self.mha_channel = MHAChannel(
                latent_dim, num_heads=num_heads, **mha_channel_kwargs
            )

        # make sure some channel is selected
        num_channels = sum([spatial_channel, lin_channel, latent_channel, mha_channel])
        assert num_channels > 0, "No channels selected"

        self.fusion = fusion if fusion is not None else ""
        self.fusion_component = FusionComponent(latent_dim, fusion=self.fusion)

        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = FeedForwardBlock(
                latent_dim, ff_hidden_dim, activation=activation, dropout=dropout
            )

        self.activation = activation_resolver(activation)
        if norm is None:
            self.norm1 = Identity()
            self.norm2 = Identity()
            self.norm3 = Identity()
        else:
            self.norm1 = normalization_resolver(norm, in_channels=latent_dim)
            self.norm2 = normalization_resolver(norm, in_channels=latent_dim)
            self.norm3 = normalization_resolver(norm, in_channels=latent_dim)

        self.plain_last = plain_last
        self.norm_first = norm_first
        self.norm_last = norm_last
        self.use_sparse = use_sparse

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        edge_attr=None,
        pos=None,
        batch=None,
        # TODO: only provide pos encoding not combined with x
        x_spatial=None,  # combined with positional encoding
    ):
        # x.shape = (batch*num_nodes, latent_dim)
        if self.norm_first:
            x = self.norm1(x)

        # TODO: check if conv takes edge weights or edge attributes
        channels = []
        if self.spatial_channel is not None:
            if x_spatial is None:
                x_spatial = x
            if self.use_sparse:
                adj = SparseTensor(
                    row=edge_index[0], col=edge_index[1], value=edge_attr
                )
                out_spatial = self.spatial_channel(x_spatial, adj.t())
            else:
                out_spatial = self.spatial_channel(
                    x_spatial, edge_index, edge_attr=edge_attr
                )
            channels.append(out_spatial)
            # might need projection back to latent dim
        if self.lin_channel is not None:
            channels.append(self.lin_channel(x))
        if self.latent_channel is not None:
            # might need projection back to latent dim
            channels.append(self.latent_channel(x, batch))
        if self.mha_channel is not None:
            channels.append(self.mha_channel(x))

        # fuse channels
        out_fusion = self.fusion_component(x, *channels)

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

        if self.norm_last:
            out = self.norm2(out)

        if not self.plain_last:
            out = self.activation(out)

        return out
    

# maybe implement extra class for spatial channel and latent channel
class GNNChannel(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers=1,
        edge_dim=1,
        activation=None,
        norm=None,
        dropout=0.0,
        residuals=False,
        **conv_kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    latent_dim,
                    latent_dim,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    **conv_kwargs,
                )
            )
            self.norms.append(normalization_resolver(norm, in_channels=latent_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.activations.append(activation_resolver(activation))
        self.residuals = residuals

    def forward(self, x, edge_index, edge_attr=None):
        for conv, norm, act, dropout in zip(
            self.convs, self.norms, self.activations, self.dropouts
        ):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if norm is not None:
                x = norm(x)
            if act is not None:
                x = act(x)
            x = dropout(x)
        return x


class MHAChannel(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_heads,
        num_layers=1,
        activation=None,
        norm=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=True,
        add_zero_attn=False,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                MultiHeadSelfAttention(
                    latent_dim,
                    num_heads,
                    dropout=dropout,
                    bias=bias,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                )
            )
            self.norms.append(normalization_resolver(norm, in_channels=latent_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.activations.append(activation_resolver(activation))

    def forward(self, x):
        for conv, norm, act, dropout in zip(
            self.convs, self.norms, self.activations, self.dropouts
        ):
            x = conv(x)[0]
            if norm is not None:
                x = norm(x)
            if act is not None:
                x = act(x)
            x = dropout(x)
        return x


class LatentChannel(GNNChannel):
    def __init__(
        self,
        latent_dim,
        knn=6,
        cosine=False,
        num_layers=1,
        activation=None,
        norm=None,
        dropout=0.0,
        residuals=False,
        **conv_kwargs,
    ):
        super().__init__(
            latent_dim,
            num_layers=num_layers,
            edge_dim=None,
            activation=activation,
            norm=norm,
            dropout=dropout,
            residuals=residuals,
            **conv_kwargs,
        )
        self.knn = knn
        self.cosine = cosine

    def forward(self, x, batch):
        edge_index = self.construct_latent_graph(x, batch)
        return super().forward(x, edge_index)

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
