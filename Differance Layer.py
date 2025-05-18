import torch
from torch import nn

class DifferanceLayer(nn.Module):
    """A layer that defers meaning through traces of past states."""

    def __init__(self, dim, max_defer=3, decay=0.9, gamma=0.5):
        super().__init__()
        self.dim = dim
        self.max_defer = max_defer
        self.decay = decay
        self.gamma = gamma
        self.proj = nn.Linear(dim, dim)
        self.deferred_states = []

    def forward(self, x):
        # x: [batch, seq_len, dim]
        if self.deferred_states:
            weights = [self.decay ** (len(self.deferred_states) - i)
                       for i in range(len(self.deferred_states))]
            total = sum(weights)
            past = sum(w * s for w, s in zip(weights, self.deferred_states)) / (total + 1e-6)
            out = x + self.gamma * (x - past)
        else:
            out = x

        self.deferred_states.append(x.detach())
        if len(self.deferred_states) > self.max_defer:
            self.deferred_states.pop(0)

        return self.proj(out)

    def reset_deferred(self):
        self.deferred_states = []
