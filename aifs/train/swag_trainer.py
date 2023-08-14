# Uses Apache 2.0 licensed code adapted from
# https://github.com/MilesCranmer/bnn_chaos_model/blob/master/spock_reg_model.py#L690
# https://github.com/MilesCranmer/bnn_chaos_model/blob/master/LICENSE
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig

from aifs.diagnostics.logger import get_logger
from aifs.train.trainer import GraphForecaster

LOGGER = get_logger(__name__)


class GraphForecasterSWAG(GraphForecaster):
    def __init__(
        self,
        metadata: Dict,
        config: DictConfig,
    ) -> None:
        super().__init__(metadata=metadata, config=config)

        # init SWAG tensors
        self.k_swag = config.training.swag.k
        self.c_swag = config.training.swag.c
        self.lr_swag = config.training.swag.lr

        n_params = sum(p.numel() for p in self.gnn.parameters())
        LOGGER.debug("Number of GNN parameters: %d ", n_params)

        LOGGER.debug("Registering SWAG tensors as module buffers ...")

        self.register_buffer("w_avg", torch.zeros(n_params, dtype=self.dtype, device=torch.device("cpu")), persistent=True)
        self.register_buffer("w_squared_avg", torch.zeros(n_params, dtype=self.dtype, device=torch.device("cpu")), persistent=True)
        self.register_buffer("Dhat", torch.zeros(1, dtype=self.dtype, device=torch.device("cpu")))  # to resize later?
        self.register_buffer("n_averaged", torch.zeros(1, dtype=self.dtype, device=torch.device("cpu")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def forward_swa(self, x: torch.Tensor) -> torch.Tensor:
        self.load(self.w_avg)
        return self.gnn(x)

    def forward_swag(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        new_w = self.sample_swag_weights(scale)
        self.load(new_w)
        return self.gnn(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.trainer.model.parameters(), lr=self.lr_swa)
        _scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [self.swa_params["swa_start"]], self.swa_params["swa_recording_lr_factor"]
        )
        scheduler = {"scheduler": _scheduler, "name": "swa_record_lr", "interval": "epoch"}

        return [optimizer], [scheduler]

    def load(self, p_vec: torch.Tensor) -> None:
        """Load a vector into the module's state dict."""
        cur_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        i = 0
        for key, old_p in cur_state_dict.items():
            size, shape = old_p.numel(), old_p.shape
            new_p = p_vec[i : i + size].reshape(*shape)
            new_state_dict[key] = new_p
            i += size
        self.gnn.load_state_dict(new_state_dict)

    def sample_swag_weights(self, scale: float = 1.0) -> torch.Tensor:
        """
        Sample weights using SWAG: w ~ N(w_avg, 1/2 * sigma + D * D^T / 2(K-1))
        This can be done with the following matrices:
            z_1 ~ N(0, I_d); d the number of parameters
            z_2 ~ N(0, I_K)
        Then, compute:
            w = w_avg + (1 / sqrt(2)) * sigma^(1/2) * z_1 + D * z_2 / sqrt(2 * (K-1))
        """
        with torch.no_grad():
            avg_w = self.w_avg
            avg_w2 = self.w2_avg
            D_hat_centered = self.D_hat - avg_w[:, None]
            d = avg_w.shape[0]
            z_1 = torch.randn((1, d), device=self.device)
            z_2 = torch.randn((self.k_swag, 1), device=self.device)
            sigma = torch.abs(torch.diag(avg_w2 - avg_w**2))
            w = avg_w[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 @ sigma**0.5
            w += scale * (D_hat_centered @ z_2).T / np.sqrt(2 * (self.k_swag - 1))
            w = w[0]
        return w
