# Uses Apache 2.0 licensed code adapted from
# https://github.com/MilesCranmer/bnn_chaos_model/blob/master/spock_reg_model.py#L690
# https://github.com/MilesCranmer/bnn_chaos_model/blob/master/LICENSE
from collections import OrderedDict
from typing import cast
from typing import Dict

import numpy as np
import torch
from lightning_fabric.utilities.types import LRScheduler
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import LRSchedulerConfig
from torch import nn
from torch.optim.swa_utils import SWALR

from aifs.diagnostics.logger import get_logger
from aifs.train.trainer import GraphForecaster

LOGGER = get_logger(__name__)


class SWAGForecaster(GraphForecaster):
    """
    Implements the SWA-Gaussian procedure that allows us to sample
    from a Bayesian (approximate) posterior of the optimal model weights.
    https://arxiv.org/pdf/1902.02476.pdf

    Limitations:
        - Not supported for models that contain BatchNorm layers; this can be added but needs tweaking, see the paper
        - Only supported on every epoch
        - Not supported for multiple optimizers / LR schedulers
    TODO:
        Make sure the values of D_hat are >= some small value (use torch.clamp)
    """

    def __init__(
        self,
        metadata: Dict,
        config: DictConfig,
    ) -> None:
        super().__init__(metadata=metadata, config=config)

        # init SWAG tensors
        # k: Maximum number of columns in the deviation matrix
        # lr_swag: The SWAG learning rate to use:
        #   - float: Use this value for all parameter groups of the optimizer.
        #   - List[float]: A list values for each parameter group of the optimizer.
        # swag_epoch_start: The SWAG procedure will start from the swa_epoch_start-th epoch.
        # swag_anneal_epochs: number of epochs in the annealing phase
        # swag_anneal_strategy: Specifies the annealing strategy (cosine or linear)
        # swag_device: if provided, the tensors will be stored on the given device.
        self.k_swag = config.training.swag.k
        self._k = 0
        self.c_swag = config.training.swag.c
        self.lr_swag = config.training.swag.lr
        self.swag_epoch_start = config.training.swag.epoch_start
        self.swag_anneal_epochs = config.training.swag.annealing.epochs
        self.swag_anneal_strategy = config.training.swag.annealing.strategy
        self.swag_device = config.training.swag.device

        LOGGER.debug(
            "SWAG parameters: LR = %.3e, epoch_start = %d, annealing-epochs: %d, annealing-strategy: %s, device: %s",
            self.lr_swag,
            self.swag_epoch_start,
            self.swag_anneal_epochs,
            self.swag_anneal_strategy,
            self.swag_device,
        )

        self._initialized = False
        self._latest_update_epoch = -1

        num_params = sum(p.numel() for p in self.gnn.parameters())
        LOGGER.debug("Number of GNN parameters: %d ", num_params)

        LOGGER.debug("Registering SWAG tensors as persistent module buffers ...")
        self.register_buffer("w_avg", torch.zeros(num_params, dtype=self.dtype, device=self.swag_device), persistent=True)
        self.register_buffer("w2_avg", torch.zeros(num_params, dtype=self.dtype, device=self.swag_device), persistent=True)
        self.register_buffer(
            "D_hat", torch.zeros((num_params, self.k_swag), dtype=self.dtype, device=self.swag_device), persistent=True
        )
        self.register_buffer("num_avg", torch.zeros(1, dtype=torch.long, device=self.swag_device))

        assert not any(
            isinstance(module, nn.modules.batchnorm._BatchNorm) for module in self.modules()
        ), "SWAG: models with BatchNorm layers are not supported (they need special treatment, see the SWAG paper)."

        self.save_hyperparameters()

    @property
    def swa_start(self) -> int:
        return max(self.swag_epoch_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._max_epochs - 1  # 0-based

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

    def update_swag_state(self) -> None:
        """
        Updates the SWAG state parameters: first & second moments and the D_hat matrix.
        Refer to algorithm 1 in https://arxiv.org/pdf/1902.02476.pdf
        """
        LOGGER.debug("Updating SWAG tensors -- this is update no %d ...", self.num_avg)

        w = self.flatten(self.parameters())
        w2 = torch.square(w)

        self.w_avg = self.w_avg + (w - self.w_avg) / (self.num_avg + 1)
        self.w2_avg = self.w2_avg + (w2 - self.w2_avg) / (self.num_avg + 1)

        if self._k < self.k_swag:
            self.D_hat[:, self._k] = w
            self._k = self._k + 1
        else:
            assert self._k == self.k_swag, f"Logic error! Got more valid D_hat columns {self._k} than SWAG tensors {self.k_swag}"
            # discard the first ("oldest") column of the \hat{D} matrix, replace with the "newest" weights
            self.D_hat = self.D_hat.roll(-1, dims=1)
            self.D_hat[:, -1] = w

        self.num_avg += 1

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
            D_hat_centered = self.D_hat[:, : self._k] - self.w_avg[:, None]  # include only the first self._k valid columns
            z_1 = torch.randn((1, self.w_avg.shape[0]), device=self.swag_device)
            z_2 = torch.randn((self.k_swag, 1), device=self.swag_device)
            sigma = torch.abs(torch.diag(self.w2_avg - self.w_avg**2))
            w = self.w_avg[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 @ sigma**0.5
            w += scale * (D_hat_centered @ z_2).T / np.sqrt(2 * (self.k_swag - 1))
            w = w[0]
        return w

    def on_fit_start(self) -> None:
        if len(self.optimizers()) != 1:
            LOGGER.error("SWAG currently works with 1 optimizer.")
            raise RuntimeError

        if len(self.lr_schedulers()) > 1:
            LOGGER.error("SWAG currently not supported for more than 1 lr_scheduler.")
            raise RuntimeError

        assert self.trainer.max_epochs is not None, "max_epochs == None in the trainer!"
        self._max_epochs = self.trainer.max_epochs

        if self._scheduler_state is not None:
            self._clear_schedulers()

    def on_train_epoch_start(self) -> None:
        if (not self._initialized) and (self.swa_start <= self.trainer.current_epoch <= self.swa_end):
            self._initialized = True

            optimizer = self.trainer.optimizers[0]
            if isinstance(self.lr_swag, float):
                self.lr_swag = [self.lr_swag] * len(optimizer.param_groups)

            for lr, group in zip(self.lr_swag, optimizer.param_groups):
                group["initial_lr"] = lr

            assert self.trainer.max_epochs is not None
            self._swa_scheduler = cast(
                LRScheduler,
                SWALR(
                    optimizer,
                    swa_lr=self._swa_lrs,  # type: ignore[arg-type]
                    anneal_epochs=self._annealing_epochs,
                    anneal_strategy=self._annealing_strategy,
                    last_epoch=self.trainer.max_epochs if self._annealing_strategy == "cos" else -1,
                ),
            )
            if self._scheduler_state is not None:
                # Restore scheduler state from checkpoint
                self._swa_scheduler.load_state_dict(self._scheduler_state)
            elif self.trainer.current_epoch != self.swa_start:
                # Log a warning if we're initializing after start without any checkpoint data,
                # as behaviour will be different compared to having checkpoint data.
                LOGGER.warning("SWAG is initializing after swa_start without any checkpoint data.")

            # We assert that there is only one optimizer on fit start
            default_scheduler_cfg = LRSchedulerConfig(self._swa_scheduler)
            assert default_scheduler_cfg.interval == "epoch"
            assert default_scheduler_cfg.frequency == 1

            if self.trainer.lr_scheduler_configs:
                scheduler_cfg = self.trainer.lr_scheduler_configs[0]
                if scheduler_cfg.interval != "epoch" or scheduler_cfg.frequency != 1:
                    LOGGER.warning("SWA(G) is currently only supported every epoch. Found %s", scheduler_cfg)
                LOGGER.info(
                    "Swapping scheduler `%s` for `%s`",
                    scheduler_cfg.scheduler.__class__.__name__,
                    self._swa_scheduler.__class__.__name__,
                )
                self.trainer.lr_scheduler_configs[0] = default_scheduler_cfg
            else:
                self.trainer.lr_scheduler_configs.append(default_scheduler_cfg)

        if (self.swa_start <= self.trainer.current_epoch <= self.swa_end) and (
            self.trainer.current_epoch > self._latest_update_epoch
        ):
            assert self.n_averaged is not None
            # assert self._swag_model is not None
            self.update_swag_state()
            self._latest_update_epoch = self.trainer.current_epoch

    def _clear_schedulers(self) -> None:
        # If we have scheduler state saved, clear the scheduler configs so that we don't try to
        # load state into the wrong type of schedulers when restoring scheduler checkpoint state.
        # We'll configure the scheduler and re-load its state in on_train_epoch_start.
        if self.trainer.lr_scheduler_configs:
            assert len(self.trainer.lr_scheduler_configs) == 1
            self.trainer.lr_scheduler_configs.clear()
