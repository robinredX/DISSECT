import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


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
        x = self.dropout2(x)
        return x
