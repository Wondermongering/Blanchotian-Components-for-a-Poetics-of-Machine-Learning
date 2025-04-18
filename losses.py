import torch
import torch.nn.functional as F
from torch import Tensor

def blanchot_neutral_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    label_smoothing: float = 0.1,
    neutral_factor: float = 0.5,
    disaster_threshold: float | None = 5.0,
    reduction: str = "mean"
) -> Tensor:
    """
    A Blanchot‑inspired criterion that treats error as a site of generativity.
    
    Parameters
    ----------
    logits : Tensor
        Shape (B, C). Raw, unnormalised model outputs.
    targets : Tensor
        Shape (B,). Integer class indices  [0, C‑1].
    label_smoothing : float
        ε in the smoothed target distribution. 0 ⇒ hard one‑hot.
    neutral_factor : float
        Weight given to the 'neutral' component (divergence from batch centre).
        0 ⇒ ordinary cross‑entropy. 1 ⇒ pure neutrality.
    disaster_threshold : float | None
        If the *per‑sample* cross‑entropy exceeds this value,
        the punitive term dominates (mitigates gradient explosion).
        Set to `None` to disable.
    reduction : {'mean','sum','none'}
        Aggregation across the batch.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be 2‑D (batch, classes)")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    B, C = logits.shape
    eps = 1.0 / C

    # ---------- 1. Label smoothing targets  ----------
    with torch.no_grad():
        smooth = torch.full_like(logits, label_smoothing * eps)
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + label_smoothing * eps)

    # ---------- 2. Stable log–softmax ----------
    log_probs = F.log_softmax(logits, dim=-1)                # uses log‑sum‑exp trick

    # Standard per‑sample cross‑entropy
    ce = -(smooth * log_probs).sum(dim=-1)                    # shape (B,)

    # ---------- 3. Neutral term ----------
    # distance from the batch's 'centre of error'
    ce_centre = ce.detach().mean()
    neutral = torch.abs(ce - ce_centre)

    # ---------- 4. Disaster threshold ----------
    if disaster_threshold is not None:
        disaster_mask = (ce > disaster_threshold).float()
        neutral = neutral * (1.0 - disaster_mask) + ce * disaster_mask

    # ---------- 5. Blend ----------
    loss = (1.0 - neutral_factor) * ce + neutral_factor * neutral

    # ---------- 6. Reduction ----------
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
