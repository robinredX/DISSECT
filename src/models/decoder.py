from torch_geometric.nn import MLP
import torch.nn as nn


class CelltypeDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_celltypes,
        num_layers=2,
        hidden_channels=64,
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
