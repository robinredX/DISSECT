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
from torch_geometric.nn.dense.linear import Linear as PygLinear

from src.models.components.fusion import GatingUnit
from src.models.decoder import CelltypeDecoder, GeneExpressionDecoder
from src.models.components.ffn import FeedForwardBlock
from src.models.transformer import TransformerEncoder
from src.models.encoder_st import MultiChannelGNNEncoder
from src.models.encoder_sc import SingleCellEncoder
from src.models.encoder.hybrid_encoder import HybridEncoder


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
            self.encoder = MultiChannelGNNEncoder(
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


class DissectSpatialHybrid(nn.Module):
    def __init__(
        self,
        num_celltypes,
        latent_dim,
        activation="relu",
        use_pos=True,
        use_id=False,
        encoder_kwargs={},
        decoder_kwargs={},
    ) -> None:
        super().__init__()
        self.encoder = HybridEncoder(
            latent_dim, activation=activation, **encoder_kwargs
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


class DissectHetero(nn.Module):
    def __init__(
        self,
        num_celltypes,
        num_genes,
        latent_dim,
        activation="relu",
        use_pos=False,
        st_encoder_kwargs={},
        sc_encoder_kwargs={},
        share_decoder=False,
        celltype_decoder_kwargs={},
        expr_decoder_kwargs={},
    ) -> None:
        super().__init__()
        self.use_pos = use_pos

        # shared encoder for mixed and real data
        # architecture should be the same as for the signle cell encoder
        # so that we can copy the weights
        self.encoder_real = MultiChannelGNNEncoder(
            latent_dim, activation=activation, **st_encoder_kwargs
        )
        self.encoder_sim = MultiChannelGNNEncoder(
            latent_dim, activation=activation, **sc_encoder_kwargs
        )

        self.expr_decoder_real = GeneExpressionDecoder(
            latent_dim, num_genes, activation=activation, **expr_decoder_kwargs
        )
        self.expr_decoder_sim = GeneExpressionDecoder(
            latent_dim, num_genes, activation=activation, **expr_decoder_kwargs
        )

        self.decoder_sim = CelltypeDecoder(
            latent_dim, num_celltypes, activation=activation, **celltype_decoder_kwargs
        )
        if share_decoder:
            self.decoder_real = self.decoder_sim
        else:
            self.decoder_real = CelltypeDecoder(
                latent_dim,
                num_celltypes,
                activation=activation,
                **celltype_decoder_kwargs,
            )

        self.z = {}

    def forward(
        self,
        x,
        edge_index,
        id,
        edge_weight=None,
        edge_attr=None,
        pos=None,
        batch=None,
    ):
        if pos is not None and self.use_pos:
            x = torch.cat([x, pos], dim=-1)

        g_type, encoder, decoder = self.resolve_graph_id(id)
        self.z[g_type] = encoder(
            x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            pos=pos,
            batch=batch,
        )
        out = decoder(self.z[g_type])
        return out

    def reconstruct(self, g_type):
        if g_type == "real" or g_type == "mix":
            expr_decoder = self.expr_decoder_real
        else:
            expr_decoder = self.expr_decoder_sim
        expr_recon = expr_decoder(self.z[g_type])
        return expr_recon

    def resolve_graph_id(self, id):
        if id[0][0] == 1:
            g_type = "real"
            encoder = self.encoder_real
            decoder = self.decoder_real
        elif id[0][1] == 1:
            g_type = "sim"
            encoder = self.encoder_sim
            decoder = self.decoder_sim
        else:
            g_type = "mix"
            encoder = self.encoder_real
            decoder = self.decoder_real
        return g_type, encoder, decoder

    def exchange_weights(self):
        copy_weights_between_nets(self.encoder_real, self.encoder_sim)
        copy_weights_between_nets(self.decoder_real, self.decoder_sim)


def copy_weights_between_nets(target_net, source_net):
    with torch.no_grad():
        for target, source in zip(target_net.parameters(), source_net.parameters()):
            target.copy_(source)
