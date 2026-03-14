import torch.nn as nn
from torchvision import models

NUM_CLASSES = 169


def build_efficientnet_b0(num_classes=NUM_CLASSES):
    """
    EfficientNet-B0 fine-tuned for mushroom classification.

    Why EfficientNet-B0?
    --------------------
    YOLO's backbone (C2f blocks) was designed for multi-scale object *detection*,
    carrying that architectural heritage even when used as a pure classifier.
    EfficientNet was designed from the ground up for classification using a compound
    scaling law — balancing depth, width, and resolution together — which makes it
    more parameter-efficient for this specific task.

    Expected improvement over YOLO26n-cls:
        - Similar model size (~5MB vs 3.5MB)
        - ~2-4% higher top-1 accuracy on 169 classes

    Architecture changes from stock EfficientNet-B0:
        - Replace final Linear(1280 → 1000) with Linear(1280 → 169)
        - Add Dropout(0.3) before the classifier to combat overfitting on 169 fine-grained classes
        - Keep ALL backbone weights (ImageNet pretrained) — only the classifier is random-init

    Input:  (batch, 3, 224, 224)
    Output: (batch, 169)  — raw logits, pass through softmax for probabilities
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # The stock classifier is: [Dropout(0.2), Linear(1280, 1000)]
    # We replace it with our own head for 169 mushroom species.
    in_features = model.classifier[1].in_features  # 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model
