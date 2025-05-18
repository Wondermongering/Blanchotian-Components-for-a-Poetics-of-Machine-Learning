import torch
from typing import List, Optional
import matplotlib.pyplot as plt

import importlib.machinery
import importlib.util

# Dynamically load BlanchotianAttention from the file with spaces
_module_name = "blanchotian_attention"
_loader = importlib.machinery.SourceFileLoader(
    _module_name, "Blanchotian Attention Mechanism.py")
_spec = importlib.util.spec_from_loader(_module_name, _loader)
_attn_module = importlib.util.module_from_spec(_spec)
_loader.exec_module(_attn_module)

BlanchotianAttention = _attn_module.BlanchotianAttention

def plot_void_attention(
    attn: BlanchotianAttention,
    x: torch.Tensor,
    token_labels: Optional[List[str]] = None,
    *,
    head: int = 0,
):
    """Plot attention including the void token.

    Parameters
    ----------
    attn : BlanchotianAttention
        Attention module instance.
    x : torch.Tensor
        Input of shape ``(B, N, D)``.
    token_labels : list[str] | None
        Optional labels for the ``N`` tokens and the void token.
    head : int
        Which head to visualise.
    """
    attn.eval()
    with torch.no_grad():
        _, attn_weights = attn(x, return_attention=True)

    # attn_weights: (B, H, N+1, N+1)
    attn_avg = attn_weights.mean(dim=0)[head]
    n = x.shape[1]
    labels = token_labels or [f"t{i}" for i in range(n)] + ["<void>"]
    if len(labels) != n + 1:
        raise ValueError("token_labels must have length N + 1")

    fig, ax = plt.subplots()
    im = ax.imshow(attn_avg.cpu().numpy(), cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
