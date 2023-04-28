from typing import Optional

import torch
from torch import nn

from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


class WeightedMSELoss(nn.Module):
    """Latitude-weighted MSE loss"""

    def __init__(self, area_weights: torch.Tensor, data_variances: Optional[torch.Tensor] = None) -> None:
        """
        Latitude- and (inverse-)variance-weighted MSE Loss.
        Args:
            area_weights: area weights
            data_variances: precomputed, per-variable stepwise variance estimate
                            V_{i,t} = E_{i,t} [ x^{t+1} - x^{t} ] (i = lat/lon index, t = time index, x = predicted variable)
        """
        super().__init__()

        self.register_buffer("weights", area_weights, persistent=True)
        if data_variances is not None:
            self.register_buffer("ivar", data_variances, persistent=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, squash=True) -> torch.Tensor:
        """
        Calculates the lat-weighted MSE loss.
        Args:
            pred: Prediction tensor, shape (bs, lat*lon, n_outputs)
            target: Target tensor, shape (bs, lat*lon, n_outputs)
        """
        if hasattr(self, "ivar"):
            if squash:
                out = (torch.square(pred - target) * self.ivar).mean(dim=-1)
            else:
                out = torch.square(pred - target) * self.ivar
        else:
            if squash:
                out = torch.square(pred - target).mean(dim=-1)
            else:
                out = torch.square(pred - target)

        if squash:
            out = out * self.weights.expand_as(out)
            out /= torch.sum(self.weights.expand_as(out))
            return out.sum()
        else:
            out = out * self.weights[..., None].expand_as(out)
            out /= torch.sum(self.weights[..., None].expand_as(out))
            return out.sum(axis=(0, 1))
