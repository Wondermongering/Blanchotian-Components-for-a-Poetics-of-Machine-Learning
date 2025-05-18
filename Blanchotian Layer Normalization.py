import torch
from torch import nn

class BlanchotianLayerNorm(nn.Module):
    """Layer normalisation preserving 'essential solitude'."""
    def __init__(self, normalized_shape, eps=1e-5, solitude_factor=2.0):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.solitude_factor = solitude_factor
        self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + self.eps)
        normalized = (x - mean) / std

        dev = torch.abs(x - mean)
        gate = torch.exp(-((dev / (self.solitude_factor * std + self.eps)) ** 2))

        blended = gate * normalized + (1.0 - gate) * x
        return blended * self.weight + self.bias
