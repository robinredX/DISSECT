import torch.nn as nn
import torch


class GatingUnit(nn.Module):
    def __init__(self, on_last_latent=True) -> None:
        super().__init__()
        self.on_last_latent = on_last_latent
        self.linear = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, *sup_channels):
        # x.shape = (batch*num_nodes, latent_dim)
        latent_dim = x.shape[-1]
        channel_cat = torch.cat([x, *sup_channels], dim=-1)
        # channel_input.shape = (batch*num_nodes, latent_dim*num_channels)
        channel_cat = channel_cat.reshape(channel_cat.shape[0], -1, latent_dim)
        num_sub_channels = channel_cat.shape[1]

        # init linear layer in first run and infer all relevant dimensions
        if self.linear is None:
            self.linear = nn.LazyLinear(num_sub_channels).to(x.device)

        # compute weights based on previouse latent representation
        if self.on_last_latent:
            logits = self.linear(x)
        else:
            logits = self.linear(channel_cat)
        weights = self.softmax(logits)
        # weights.shape = (batch*num_nodes, num_sub_channels)
        weights = weights.reshape(weights.shape[0], num_sub_channels, 1)
        # weights.shape = (batch*num_nodes, num_sub_channels, 1)

        # compute weighted sum
        out = torch.sum(weights * channel_cat, dim=1)
        # out.shape = (batch*num_nodes, latent_dim)
        return out
