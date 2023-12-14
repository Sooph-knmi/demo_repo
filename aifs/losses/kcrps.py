from typing import Optional

import torch
from torch import nn

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=False)


class KernelCRPS(nn.Module):
    """Area-weighted kernel CRPS loss."""

    def __init__(self, area_weights: torch.Tensor, loss_scaling: Optional[torch.Tensor] = None, fair: bool = True) -> None:
        """
        Latitude- and (inverse-)variance-weighted kernel CRPS loss.
        Args:
            area_weights: area weights
            loss_scaling: weight loss components to ensure all vars contribute ~ equally to the total value
            fair: calculate a "fair" (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1
        """
        super().__init__()

        self.fair = fair
        self.register_buffer("weights", area_weights, persistent=True)
        if loss_scaling is not None:
            self.register_buffer("scale", loss_scaling, persistent=True)

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, fair: bool = True) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Args:
            targets: ground truth, shape (batch_size, n_vars, latlon)
            preds: predicted ensemble, shape (batch_size, n_vars, latlon, ens_size)
            fair: unbiased ensemble variance calculation
        Returns:
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """
        ens_size = preds.shape[-1]
        mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

        if ens_size == 1:
            # mse = torch.square(targets[..., None] - preds)[:, :, :, 0]
            return mae

        if fair:
            coef = -1.0 / (2.0 * ens_size * (ens_size - 1))
        else:
            coef = -1.0 / (2.0 * ens_size**2)
        ens_var = coef * torch.sum(torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2)), dim=(-1, -2))
        return mae + ens_var

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, squash: bool = True) -> torch.Tensor:
        bs_ = y_pred.shape[0]  # batch size
        kcrps_ = self._kernel_crps(y_pred, y_target, fair=self.fair)
        if hasattr(self, "scale"):
            kcrps_ = kcrps_ * self.scale[:, None]
        if y_pred.shape[-1] == 1 and squash:
            kcrps_ = kcrps_.mean(dim=1)
            kcrps_ = kcrps_ * self.weights.expand_as(kcrps_)
            kcrps_ /= torch.sum(self.weights.expand_as(kcrps_))
            return kcrps_.sum()
        else:
            kcrps_ = kcrps_ * self.weights
            # divide by (weighted point count) * (batch size)
            npoints = torch.sum(self.weights)
            if squash:
                return kcrps_.sum() / (npoints * bs_)
            # sum only across the batch dimension; enable this to generate per-variable CRPS "maps"
            return kcrps_.sum(dim=0) / bs_
