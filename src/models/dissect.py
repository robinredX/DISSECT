import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from torch_geometric.nn.inits import glorot, zeros


class Dissect(nn.Module):
    def __init__(
        self,
        num_genes,
        num_celltypes,
        channel_list=[512, 256, 128, 64],
        act="relu6",
        use_pos=False,
    ) -> None:
        super().__init__()
        if not use_pos:
            channel_list = [num_genes] + channel_list + [num_celltypes]
        else:
            channel_list = [num_genes + 2] + channel_list + [num_celltypes]
        self.mlp = self.mlp = MLP(
            channel_list,
            norm=None,
            plain_last=True,
            act=act,
        )
        self.softmax = nn.Softmax(dim=1)
        self.use_pos = use_pos
        self.reset_parameters()

    def forward(
        self, x, edge_index=None, edge_weight=None, edge_attr=None, pos=None, batch=None
    ):
        if pos is not None and self.use_pos:
            x = torch.cat([x, pos], dim=-1)
        x = self.mlp(x)
        x = self.softmax(x)
        return x

    def reset_parameters(self):
        for w in self.parameters():
            if len(w.shape) == 1:
                zeros(w)
            else:
                glorot(w)
