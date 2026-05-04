"""
DINOv2-Small model builder for mushroom classification.

Uses Facebook's DINOv2 backbone (Oquab et al., 2023) loaded from
torch.hub (facebookresearch/dinov2). A linear classification head is
appended and the full model is fine-tuned end-to-end.

Why DINOv2?
-----------
DINOv2 is trained with self-supervised learning on 142M curated images
(LVD-142M). It produces dense, spatially-aware features without ever
seeing a label — which makes its representations remarkably transferable
to niche domains like mycology.

Compared to ViT-S/16 (ImageNet-supervised):
  - DINOv2-Small has the *same* ViT-S/16 architecture (384-dim, 6 heads)
  - But was pretrained with self-distillation across patch tokens, giving
    richer local features (gill texture, cap shape details)
  - Consistently outperforms supervised ViT on fine-grained benchmarks

Architecture: ViT-S/14 (14×14 patches, 384 hidden dim, 21M params)
Pretrained: LVD-142M curated web images (self-supervised DINO v2 objective)

Requirements
------------
    pip install timm  (already in requirements.txt)
    # DINOv2 is loaded via torch.hub — no extra install needed
"""

import torch
import torch.nn as nn

NUM_CLASSES = 169


class DINOv2Classifier(nn.Module):
    """
    DINOv2-Small backbone + linear classification head.

    The backbone produces a [CLS] token embedding of dimension 384.
    We attach a single Linear(384 → num_classes) head — the simplest
    possible head, which is standard practice for DINOv2 linear probing
    and fine-tuning benchmarks.

    Fine-tuning strategy (implemented in train.py):
      - Backbone: lower LR (0.1× of head LR) — preserves rich pretrained features
      - Head: full LR — randomly initialised, needs faster updates
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Load DINOv2-Small from facebookresearch hub
        # dinov2_vits14: ViT-Small with 14×14 patches, 384-dim CLS token
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
        )
        embed_dim = self.backbone.embed_dim  # 384 for ViT-Small
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 returns the [CLS] token features directly
        features = self.backbone(x)   # (batch, 384)
        return self.head(features)    # (batch, num_classes)


def build_dinov2_small(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Return a DINOv2-Small classifier ready for fine-tuning.

    Input:  (batch, 3, 224, 224)  — ImageNet-normalised tensor
    Output: (batch, num_classes)  — raw logits
    """
    return DINOv2Classifier(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = build_dinov2_small()
    print(f"DINOv2-Small — {count_parameters(m):,} parameters")
    dummy = torch.randn(2, 3, 224, 224)
    out = m(dummy)
    print(f"Output shape: {out.shape}")  # (2, 169)
