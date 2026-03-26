"""
TaxonomicLoss — weighted combination of genus and species cross-entropy.

Why weight them differently?
-----------------------------
Species prediction is the primary objective, so it gets the larger weight
(beta).  Genus prediction is auxiliary — it's a simpler task that adds a
regularising signal to the shared backbone without dominating the gradient.

    total_loss = alpha * CE(genus_logits, genus_labels)
               + beta  * CE(species_logits, species_labels)

Default weights  alpha=0.3, beta=0.7  sum to 1.0 so the total loss
magnitude stays comparable to a single-head model, making LR tuning easier.
"""

import torch
import torch.nn as nn


class TaxonomicLoss(nn.Module):
    """
    Parameters
    ----------
    alpha : float   weight for the genus loss term    (default 0.3)
    beta  : float   weight for the species loss term  (default 0.7)
    label_smoothing : float   applied to both CE losses (default 0.1)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self._genus_ce   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._species_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        genus_logits:   torch.Tensor,
        species_logits: torch.Tensor,
        genus_labels:   torch.Tensor,
        species_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss    : scalar tensor (alpha * genus + beta * species)
        genus_loss    : scalar tensor — logged separately for diagnostics
        species_loss  : scalar tensor — logged separately for diagnostics
        """
        genus_loss   = self._genus_ce(genus_logits,   genus_labels)
        species_loss = self._species_ce(species_logits, species_labels)
        total_loss   = self.alpha * genus_loss + self.beta * species_loss
        return total_loss, genus_loss, species_loss
