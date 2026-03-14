"""
Custom loss functions for mushroom CNN experiments.

Why not just CrossEntropy?
--------------------------
CrossEntropyLoss treats every wrong answer the same. If the model is 99% confident
on an easy sample it still gets penalised just as much as a hard sample it was
45% confident on. FocalLoss fixes this by suppressing the easy samples so the model
spends more gradient budget on the ones it struggles with — useful here because some
mushroom species have far fewer images than others.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017 — originally for object detection).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0  → identical to standard CrossEntropyLoss
    gamma=2  → the paper's recommended default, heavily down-weights easy samples

    label_smoothing is applied before the focal weighting so you can combine both.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)

        # Standard cross-entropy with optional label smoothing
        ce_loss = F.cross_entropy(
            inputs, targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # p_t = probability assigned to the correct class
        with torch.no_grad():
            p_t = torch.exp(-ce_loss)

        # Focal weight — suppress confident (easy) samples
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(name: str) -> nn.Module:
    """
    Factory — return a loss function by name.

    Available names
    ---------------
    "ce"          CrossEntropyLoss (hard labels)
    "ce_smooth"   CrossEntropyLoss + label_smoothing=0.1
    "focal"       FocalLoss(gamma=2)
    "focal_smooth" FocalLoss(gamma=2) + label_smoothing=0.1
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
    Factory — return an optimizer by name.

    Available names
    ---------------
    "adamw"  AdamW — fast convergence, good default for fine-tuning
    "sgd"    SGD + Nesterov momentum=0.9 — slower but often better final minimum
    "radam"  RAdam — Rectified Adam, fixes variance issues in early training
             (no LR warmup needed, unlike vanilla Adam)
    """
    if name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        # SGD needs a higher LR than Adam — 0.01 is a common transfer-learning default
        return torch.optim.SGD(
            model_params, lr=lr * 10, momentum=0.9,
            weight_decay=weight_decay, nesterov=True,
        )
    elif name == "radam":
        return torch.optim.RAdam(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: adamw, sgd, radam")
