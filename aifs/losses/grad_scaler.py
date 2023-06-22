from typing import Optional
from typing import Tuple

import torch
from torch import nn


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
