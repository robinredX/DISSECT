import torch.nn as nn


class GatingUnit(nn.Module):
    def __init__(self, num_channels) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.linear = nn.LazyLinear(num_channels + 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, *channels):
        # compute weights based on previouse latent representation
        logits = self.linear(x)
        weights = self.softmax(logits)
        # compute weighted sum
        out = weights[:, [0]] * x
        for i in range(1, self.num_channels + 1):
            out += weights[:, [i]] * channels[i]
        return out
