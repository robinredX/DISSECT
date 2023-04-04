import torch
import torch.nn as nn
from torch_geometric.nn import MLP

from src.models.transformer import TransformerBlock
from src.models.encoder_st import MultiChannelGNNBlock


class HybridEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_gnn_layers=1,
        num_transformer_layers=1,
        activation="relu6",
        dropout=0.0,
        norm=None,
        num_heads=1,
        spatial_conv_kwargs={},
        ff_hidden_dim=256,
        bias=True,
        skip_connections=True,
        init_embed_hidden_channels=[512, 256],
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            [-1, *init_embed_hidden_channels, latent_dim],
            norm=None,
            plain_last=True,
            act=activation,
        )
        self.gnn_layers = nn.ModuleList()
        for layer in range(num_gnn_layers):
            self.gnn_layers.append(
                MultiChannelGNNBlock(
                    latent_dim,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    lin_channel=False,
                    spatial_channel=True,
                    num_heads=num_heads,
                    spatial_conv_kwargs=spatial_conv_kwargs,
                    latent_channel=False,
                    fusion="concat",
                    use_ffn=False,
                    ff_hidden_dim=256,
                    plain_last=True,
                )
            )
        self.transformer_layers = nn.ModuleList()
        for layer in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerBlock(
                    latent_dim,
                    activation=activation,
                    num_heads=num_heads,
                    dropout=dropout,
                    norm=norm,
                    use_ffn=True,
                    ff_hidden_dim=ff_hidden_dim,
                    bias=bias,
                )
            )
        self.skip_connections = skip_connections

    def forward(
        self, x, edge_index, edge_weight=None, edge_attr=None, pos=None, batch=None
    ):
        x = self.mlp(x)
        # gnn stack first consider as positional encoding
        x = x + self.gnn_embed(x, edge_index, edge_weight, edge_attr, pos, batch)

        # transformer stack next
        for layer in self.transformer_layers:
            if self.skip_connections:
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def gnn_embed(
        self, x, edge_index, edge_weight=None, edge_attr=None, pos=None, batch=None
    ):
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_weight, edge_attr, pos, batch)
        return x
