from typing import Optional

import einops
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric import data


class EnergyScore(nn.Module):
    def __init__(self, area_weights: torch.Tensor, loss_scaling: Optional[torch.Tensor] = None) -> None:
        """Energy score."""
        super().__init__()
        self.register_buffer("weights", area_weights, persistent=True)

        if loss_scaling is not None:
            self.register_buffer("scale", loss_scaling, persistent=True)

    def _calc_energy_score(self, preds: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
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

        m = preds.shape[1]  # ensemble size

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

    def _energy_score(self, preds: torch.Tensor, target: torch.Tensor, graph_data: data, beta: float = 1.0) -> torch.Tensor:
        preds = einops.rearrange(preds, "bs m v latlon -> bs m (latlon v)")
        target = einops.rearrange(target, "bs v latlon -> bs (latlon v)")
        energy_score = self._calc_energy_score(preds, target, beta)

        return energy_score

    def forward(self, preds: torch.Tensor, target: torch.Tensor, graph_data: data, beta: float = 1.0) -> torch.Tensor:
        preds = (preds * self.scale[None, None, :, None]) * self.weights
        target = (target * self.scale[None, :, None]) * self.weights
        return self._energy_score(preds, target, graph_data, beta)


class PatchedEnergyScore(EnergyScore):
    def __init__(self, area_weights: torch.Tensor, loss_scaling: Optional[torch.Tensor] = None) -> None:
        """Energy score."""
        super().__init__(area_weights, loss_scaling)
        self.register_buffer("weights", area_weights, persistent=True)

        if loss_scaling is not None:
            self.register_buffer("scale", loss_scaling, persistent=True)

        patches = np.load("aifs/losses/patches_o96.npy")
        self.register_buffer("patches", torch.Tensor(patches), persistent=True)

    def mask_score(self, index, masks, preds, target, beta):
        mask = masks[index].to(device="cuda")
        preds_masked_reshape = einops.rearrange(preds * mask, "bs m v latlon -> bs m (latlon v)")
        target_masked_reshape = einops.rearrange(target * mask, "bs v latlon -> bs (latlon v)")

        return self._calc_energy_score(preds_masked_reshape, target_masked_reshape, beta)

    def _patched_energy_score(self, preds: torch.Tensor, target: torch.Tensor, graph_data: data, beta: float = 1.0) -> torch.Tensor:
        preds.shape[1]  # ensemble size

        masks = self.patches

        energy_score = 0

        for index in range(masks.shape[0]):
            energy_value = checkpoint(self.mask_score, index, masks, preds, target, beta, use_reentrant=False)

            energy_score += energy_value

        return energy_score

    def forward(self, preds: torch.Tensor, target: torch.Tensor, graph_data: data, beta: float = 1.0) -> torch.Tensor:
        preds = (preds * self.scale[None, None, :, None]) * self.weights
        target = (target * self.scale[None, :, None]) * self.weights
        return self._patched_energy_score(preds, target, graph_data, beta)
