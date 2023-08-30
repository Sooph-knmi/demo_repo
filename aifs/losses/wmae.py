from typing import Optional

import torch
from torch import nn

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class WeightedMAELoss(nn.Module):
    """Weighted MAE loss."""

    def __init__(self, area_weights: torch.Tensor, loss_scaling: Optional[torch.Tensor] = None) -> None:
        """Area-weighted + component-scaled MAE Loss.

        Args:
            area_weights: area weights
            loss_scaling: weight loss components to ensure all vars contribute ~ equally to the total value
        """
        super().__init__()

        self.register_buffer("weights", area_weights, persistent=True)
        if loss_scaling is not None:
            self.register_buffer("scale", loss_scaling, persistent=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, squash: bool = True) -> torch.Tensor:
        """Calculates the area-weighted MAE loss.

        Args:
            pred: Prediction tensor, shape (bs, lat*lon, n_outputs)
            target: Target tensor, shape (bs, lat*lon, n_outputs)
            squash: if False, return a (n_outputs, 1) tensor with the individual loss contributions
                    if True, return the (scalar) total loss
        """
        if hasattr(self, "scale"):
            if squash:
                out = (torch.abs(pred - target) * self.scale).mean(dim=-1)
            else:
                out = torch.abs(pred - target) * self.scale
        else:
            if squash:
                out = torch.abs(pred - target).mean(dim=-1)
            else:
                out = torch.abs(pred - target)

        if squash:
            out = out * self.weights.expand_as(out)
            out /= torch.sum(self.weights.expand_as(out))
            return out.sum()

        out = out * self.weights[..., None].expand_as(out)
        out /= torch.sum(self.weights[..., None].expand_as(out))
        return out.sum(axis=(0, 1))
