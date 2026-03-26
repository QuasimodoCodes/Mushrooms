"""
TaxonomicYOLO26 — two-headed classifier built on the YOLO26n-cls backbone.

Architecture
------------
    YOLO26n-cls backbone  (all layers except the final Classify layer)
        ↓  global-average-pooled feature vector  (dim probed at init)
    ┌───┴────────────────┐
    genus_head           species_head
    Linear(feat → G)     Linear(feat → S)

Why strip only the final layer?
    The Classify layer in Ultralytics is just nn.Linear + flatten — it holds
    no spatial knowledge.  Everything before it is the learned visual feature
    extractor we want to reuse.

Why two separate heads instead of one?
    Genus prediction is a strictly easier task (fewer classes, coarser
    boundary).  Giving it a dedicated head lets it converge faster and
    provides a gradient signal that regularises the shared backbone,
    encouraging it to learn features that generalise across the taxonomy
    rather than over-specialising on leaf-level species differences.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO


class TaxonomicYOLO26(nn.Module):
    """
    Parameters
    ----------
    num_genera  : int   number of unique genus classes
    num_species : int   number of unique species classes
    weights     : str   ultralytics model tag or local .pt path
    """

    def __init__(self, num_genera: int, num_species: int,
                 weights: str = "yolo26n-cls.pt"):
        super().__init__()

        # Load the full classification model via ultralytics then discard the head.
        yolo = YOLO(weights)
        self._inner = yolo.model   # ultralytics ClassificationModel (nn.Module)

        # Probe the feature dimension BEFORE we do anything else, so that
        # genus_head / species_head are sized correctly even if the backbone
        # changes between YOLO versions.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat  = self._extract_features(dummy)
            feat_dim = feat.shape[1]

        self.genus_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feat_dim, num_genera),
        )
        self.species_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feat_dim, num_species),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manually replicate the ultralytics forward pass but stop one layer
        before the terminal Classify module, then flatten to a 1-D vector.

        This mirrors _forward_once() in the Ultralytics source so that skip
        connections (m.f != -1) are handled correctly.
        """
        layers = list(self._inner.model)   # ModuleList of all blocks
        backbone_layers = layers[:-1]      # everything except the Classify head

        y = []   # cache of per-layer outputs for skip connections
        for m in backbone_layers:
            if m.f != -1:
                # Skip-connection: pull from saved outputs
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            x = m(x)
            y.append(x if m.i in self._inner.save else None)

        # Flatten spatial dims → 1-D feature vector per sample.
        # Handles both 4-D (B, C, H, W) and already-pooled 2-D (B, C) outputs.
        if isinstance(x, (list, tuple)):
            x = torch.cat([t.flatten(1) for t in x], dim=1)
        else:
            x = x.flatten(1)

        return x   # shape: (batch, feat_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        genus_logits   : Tensor (batch, num_genera)
        species_logits : Tensor (batch, num_species)
        """
        feat = self._extract_features(x)
        return self.genus_head(feat), self.species_head(feat)
