import torch
import pytest


# directly import blanchot_neutral_loss from losses.py
from losses import blanchot_neutral_loss


def test_blanchot_neutral_loss_output_shape_mean():
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss = blanchot_neutral_loss(logits, targets, reduction="mean")
    assert loss.dim() == 0


def test_blanchot_neutral_loss_output_shape_none():
    logits = torch.randn(5, 7)
    targets = torch.randint(0, 7, (5,))
    loss = blanchot_neutral_loss(logits, targets, reduction="none")
    assert loss.shape == torch.Size([5])


def test_blanchot_neutral_loss_invalid_logits_dim():
    logits = torch.randn(2, 3, 4)
    targets = torch.randint(0, 4, (2,))
    with pytest.raises(ValueError):
        blanchot_neutral_loss(logits, targets)


def test_blanchot_neutral_loss_invalid_reduction():
    logits = torch.randn(3, 4)
    targets = torch.randint(0, 4, (3,))
    with pytest.raises(ValueError):
        blanchot_neutral_loss(logits, targets, reduction="avg")
