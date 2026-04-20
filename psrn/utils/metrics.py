"""Metrics and loss computation utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_losses(
    outputs: dict[str, Tensor],
    targets: Tensor,
    weight_decay: float = 4e-5,
    model: torch.nn.Module | None = None,
) -> dict[str, Tensor]:
    """Compute PSRN losses.
    
    From Eq. (9)-(10): L_total = L_pos + L_vel + L_rel + λ‖θ‖²
    
    Args:
        outputs: dict with 'pos', 'vel', 'rel' logits (B, C)
        targets: (B,) class indices
        weight_decay: weight decay coefficient λ
        model: model for weight decay computation
    
    Returns:
        dict with 'pos', 'vel', 'rel', 'total' losses
    """
    criterion = torch.nn.CrossEntropyLoss()

    loss_pos = criterion(outputs["pos"], targets)
    loss_vel = criterion(outputs["vel"], targets)
    loss_rel = criterion(outputs["rel"], targets)

    # Weight decay
    wd_loss = torch.tensor(0.0, device=targets.device)
    if model is not None and weight_decay > 0:
        for param in model.parameters():
            if param.requires_grad:
                wd_loss += weight_decay * torch.sum(param ** 2)

    loss_total = loss_pos + loss_vel + loss_rel + wd_loss

    return {
        "pos": loss_pos,
        "vel": loss_vel,
        "rel": loss_rel,
        "total": loss_total,
    }


def compute_accuracy(outputs: dict[str, Tensor], targets: Tensor) -> dict[str, float]:
    """Compute accuracy for each stream.
    
    Args:
        outputs: dict with 'pos', 'vel', 'rel' logits (B, C)
        targets: (B,) class indices
    
    Returns:
        dict with 'pos', 'vel', 'rel' accuracies (0-1)
    """
    accuracies = {}
    for stream_name, logits in outputs.items():
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        accuracies[stream_name] = acc

    return accuracies

