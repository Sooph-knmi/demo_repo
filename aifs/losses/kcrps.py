from typing import Optional

import torch
from torch import nn

from aifs.diagnostics.logger import get_logger

LOGGER = get_logger(__name__, debug=False)


class KernelCRPS(nn.Module):
    """Area-weighted kernel CRPS loss."""

    def __init__(self, area_weights: torch.Tensor, loss_scaling: Optional[torch.Tensor] = None, fair: bool = False) -> None:
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

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, fair: bool = False) -> torch.Tensor:
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
        if fair:
            # Note (Mat): do not train only on this loss
            # Best to use in combination with another loss, like WMSE or MS_SSIM
            coef = -1.0 / (2.0 * ens_size * (ens_size - 1))
        else:
            coef = -1.0 / (2.0 * ens_size**2)
        ens_var = coef * torch.sum(torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2)), dim=(-1, -2))
        return mae + ens_var

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, reduce_sum: bool = True) -> torch.Tensor:
        bs_ = y_pred.shape[0]  # batch size
        kcrps_ = self._kernel_crps(y_pred, y_target, fair=self.fair)
        LOGGER.debug(
            "kcrps_.shape = %s, scale[:, None].shape = %s, weights.shape = %s",
            kcrps_.shape,
            self.scale[:, None].shape,
            self.weights.shape,
        )
        kcrps_ = (kcrps_ * self.scale[:, None]) * self.weights
        # divide by (weighted point count) * (batch size)
        npoints = torch.sum(self.weights)
        if reduce_sum:
            return kcrps_.sum() / (npoints * bs_)
        # sum only across the batch dimension; useful when looking to plot CRPS "maps"
        return kcrps_.sum(dim=0) / bs_
