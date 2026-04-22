"""
ConvNeXt-Tiny model builder for mushroom classification.

Uses torchvision (already a project dependency — no extra installs needed).

Why ConvNeXt-Tiny?
------------------
ConvNeXt (Liu et al., 2022 — "A ConvNet for the 2020s") modernises the
standard ResNet by adopting ideas from Vision Transformers:

  • 7×7 depthwise convolutions  (vs 3×3 in ResNet)
  • Inverted bottleneck design  (wider → narrow → wide, like ViT FFN)
  • LayerNorm instead of BatchNorm
  • GELU activation instead of ReLU
  • Fewer activation + norm layers per block

The result is a pure-CNN that matches or exceeds ViT performance while
keeping the inductive biases (locality, translation equivariance) that make
CNNs efficient on limited data. ConvNeXt-Tiny should comfortably beat
EfficientNet-B0 (89.5% top-1) at similar parameter count.

Architecture head change
------------------------
Stock head: LayerNorm → Flatten → Linear(768 → 1000)
Our head  : LayerNorm → Flatten → Dropout(0.3) → Linear(768 → 169)

The Dropout reduces overfitting on 169 fine-grained species — same strategy
used in the existing EfficientNet-B0 model.

Input:  (batch, 3, 224, 224)
Output: (batch, 169)  — raw logits
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 169


def build_convnext_tiny(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Return ImageNet-pretrained ConvNeXt-Tiny with a custom mushroom head.
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # Stock classifier: Sequential(LayerNorm2d, Flatten, Linear(768, 1000))
    in_features = model.classifier[2].in_features  # 768

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = build_convnext_tiny()
    print(f"ConvNeXt-Tiny — {count_parameters(m):,} parameters")
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print(f"Output shape: {out.shape}")   # (2, 169)
