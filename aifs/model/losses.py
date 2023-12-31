from typing import Optional
from typing import Tuple

import torch
from torch import nn

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def grad_scaler(
    module: nn.Module, grad_in: Tuple[torch.Tensor, ...], grad_out: Tuple[torch.Tensor, ...]
) -> Optional[Tuple[torch.Tensor, ...]]:
    """
    Scales the loss gradients using the formula in https://arxiv.org/pdf/2306.06079.pdf, section 4.3.2
    Args:
        module: nn.Module (the loss object, not used)
        grad_in: input gradients
        grad_out: output gradients (not used)
    Returns:
        Re-scaled input gradients
    Use <module>.register_full_backward_hook(grad_scaler, prepend=False) to register this hook.
    """
    del module, grad_out  # not needed
    # loss = module(x_pred, x_true)
    # so - the first grad_input is that of the predicted state and the second is that of the "ground truth" (== zero)
    C = grad_in[0].shape[-1]  # number of channels
    w_c = torch.reciprocal(torch.sum(torch.abs(grad_in[0]), dim=1, keepdim=True))  # channel-wise weights
    new_grad_in = (C * w_c) / torch.sum(w_c, dim=-1, keepdim=True) * grad_in[0]  # rescaled gradient
    return new_grad_in, grad_in[1]


class WeightedMSELoss(nn.Module):
    """Latitude-weighted MSE loss."""

    def __init__(self, area_weights: torch.Tensor, data_variances: Optional[torch.Tensor] = None) -> None:
        """Latitude- and (inverse-)variance-weighted MSE Loss.

        Parameters
        ----------
        area_weights : torch.Tensor
            Weights by area
        data_variances : Optional[torch.Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        """
        super().__init__()

        self.register_buffer("weights", area_weights, persistent=True)
        if data_variances is not None:
            self.register_buffer("ivar", data_variances, persistent=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, squash=True) -> torch.Tensor:
        """Calculates the lat-weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True

        Returns
        -------
        torch.Tensor
            Weighted MSE loss
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

        out = out * self.weights[..., None].expand_as(out)
        out /= torch.sum(self.weights[..., None].expand_as(out))
        return out.sum(axis=(0, 1))
