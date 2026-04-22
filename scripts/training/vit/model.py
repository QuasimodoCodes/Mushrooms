"""
ViT-S/16 model builder for mushroom classification.

Uses timm (PyTorch Image Models) to load a pretrained Vision Transformer
Small/16 — the 22M-param variant that balances accuracy and speed.

Why ViT-S/16?
-------------
ViT-Small is the lightest ViT variant that still delivers strong fine-grained
classification accuracy. The "/16" means each 224×224 image is split into
(224/16)² = 196 patches of 16×16 pixels. Each patch becomes a token, and
self-attention lets every token attend to every other token — enabling the
model to correlate the gill texture in one corner with the cap shape in another,
which CNNs can only do implicitly through deep stacking.

Pretrained weights: ImageNet-21k → fine-tuned on ImageNet-1k (timm default).
The 21k pretraining exposes the backbone to far more visual concepts, which
improves transfer to niche domains like mycology.

Requirements
------------
    pip install timm
"""

import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for ViT. Install it with:  pip install timm"
    )

NUM_CLASSES = 169


def build_vit_small(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Return a ViT-S/16 with its classification head replaced for `num_classes`.

    timm handles the head replacement automatically when num_classes is passed
    to create_model — the original Linear(384 → 1000) is replaced with
    Linear(384 → num_classes).

    Input:  (batch, 3, 224, 224)  — standard ImageNet-normalised tensor
    Output: (batch, num_classes)  — raw logits
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import torch
    m = build_vit_small()
    print(f"ViT-S/16 — {count_parameters(m):,} parameters")
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print(f"Output shape: {out.shape}")   # (2, 169)
