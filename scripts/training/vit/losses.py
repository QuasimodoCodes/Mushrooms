"""
Shared loss functions and optimisers for Herman experiments.

Identical to scripts/training/cnn/losses.py so results are comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0 → equivalent to standard CrossEntropyLoss.
    gamma=2 → paper default, suppresses easy samples so the model focuses on
              hard ones. Useful here because some mushroom species have far
              fewer images than others.

    label_smoothing can be combined with focal weighting.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        with torch.no_grad():
            p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(name: str) -> nn.Module:
    """
    Return a loss function by name.

    Options: "ce" | "ce_smooth" | "focal" | "focal_smooth"
    """
    options = {
        "ce":           nn.CrossEntropyLoss(),
        "ce_smooth":    nn.CrossEntropyLoss(label_smoothing=0.1),
        "focal":        FocalLoss(gamma=2.0, label_smoothing=0.0),
        "focal_smooth": FocalLoss(gamma=2.0, label_smoothing=0.1),
    }
    if name not in options:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(options)}")
    return options[name]


def build_optimizer(name: str, model_params, lr: float, weight_decay: float):
    """
    Return an optimiser by name.

    Options: "adamw" | "sgd" | "radam"
    """
    if name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(
            model_params, lr=lr * 10, momentum=0.9,
            weight_decay=weight_decay, nesterov=True,
        )
    elif name == "radam":
        return torch.optim.RAdam(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: adamw, sgd, radam")
