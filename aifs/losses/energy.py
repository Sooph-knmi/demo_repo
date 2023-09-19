from typing import Optional

import einops
import torch
from torch import nn


class EnergyScore(nn.Module):
    def __init__(self, area_weights: torch.Tensor, loss_scaling: Optional[torch.Tensor] = None) -> None:
        """Energy score."""
        super().__init__()
        self.register_buffer("weights", area_weights, persistent=True)
        if loss_scaling is not None:
            self.register_buffer("scale", loss_scaling, persistent=True)

    def _energy_score(self, preds: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Calculates the energy score (vectorized version).
        See https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting/blob/main/src/scoring_rules.py.
        This produces the same energy score as Lorenzo's implementation, up to a constant scaling factor (= 2.0).

        Args:
            y_pred: forecast realizations, shape (bs, nvar, latlon, nens)
            y_true: ground truth ("observations"), shape (bs, nvar, latlon)
            beta: beta exponent for the energy loss (beta = 1.0 yields the CRPS of the ensemble distribution)
        Returns:
            The energy score loss.
        """
        m = preds.shape[-1]  # ensemble size

        preds = einops.rearrange(preds, "bs v latlon m -> bs m (latlon v)")
        target = einops.rearrange(target, "bs v latlon -> bs (latlon v)")

        score_a = (1.0 / m) * torch.mean(
            torch.sum(
                torch.pow(
                    torch.linalg.norm((preds - target[:, None, :]), dim=-1, ord=2),
                    beta,
                ),
                dim=-1,
            )
        )

        score_b = (
            1.0
            / (2 * m * (m - 1))
            * torch.mean(
                torch.sum(
                    torch.pow(torch.cdist(preds, preds, p=2), beta),
                    axis=(1, 2),
                ),
                dim=0,
            )
        )

        return score_a - score_b

    def forward(self, preds: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        preds = (preds * self.scale[:, None]) * self.weights
        target = (target * self.scale[:, None]) * self.weights
        return self._energy_score(preds, target, beta)
