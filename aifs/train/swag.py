from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import LRScheduler
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import LRSchedulerConfig
from torch import nn
from torch.optim.swa_utils import SWALR

from aifs.diagnostics.logger import get_logger

LOGGER = get_logger(__name__)


class SWAG(Callback):
    """Stochastic weight averaging with Gaussian uncertainty quantification.

    https://arxiv.org/pdf/1902.02476.pdf
    """

    def __init__(
        self,
        swa_lrs: Union[float, List[float]],
        swa_epoch_start: int = 1,
        k: int = 5,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        device: Optional[Union[torch.device, str]] = torch.device("cpu"),
    ) -> None:
        """
            Implements the SWA-Gaussian Callback to sample from a Bayesian approximation to the model posterior.
            https://arxiv.org/pdf/1902.02476.pdf
            Limitations:
                - Not supported for models that contain BatchNorm layers
                - Only supported on every epoch
                - Not supported for multiple optimizers / LR schedulers

        Args:
            k: Maximum number of columns in the deviation matrix
            swa_lrs: The SWA(G) learning rate to use:
                - float: Use this value for all parameter groups of the optimizer.
                - List[float]: A list values for each parameter group of the optimizer.
            swa_epoch_start: The SWAG procedure will start from the swa_epoch_start-th epoch.
            annealing_epochs: number of epochs in the annealing phase (default: 10)
            annealing_strategy: Specifies the annealing strategy (cosine or linear)
            device: If provided, the averaged model will be stored on the given device.
                When None is provided, it will infer the device from the pl.LightningModule object.
        """
        self.n_averaged: Optional[torch.Tensor] = None
        self._swa_epoch_start = swa_epoch_start
        self._swa_lrs = swa_lrs
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy

        self._device = device
        self._initialized = False
        self._swa_scheduler: Optional[LRScheduler] = None
        self._scheduler_state: Optional[Dict] = None
        self._init_n_averaged = 0
        self._latest_update_epoch = -1
        self._max_epochs: int

        # self._swag_model: Optional[pl.LightningModule] = None

        # SWAG tensors
        # See Algorithm 1 on page 5 of https://arxiv.org/pdf/1902.02476.pdf
        self._first_moment: Optional[torch.Tensor] = None  # running average of weights (first moment of the SWAG posterior)
        self._second_moment: Optional[torch.Tensor] = None  # running average of squared weights (second moment)
        self._Dhat: Optional[torch.Tensor] = None  # deviation matrix \hat{D}, D_i = (w_i - w_avg_i)
        self.k = k

    @property
    def swa_start(self) -> int:
        return max(self._swa_epoch_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._max_epochs - 1  # 0-based

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        # copy the model before moving it to accelerator device.
        # self._swag_model = deepcopy(pl_module)
        self._first_moment = pl_module.w_avg.to(self._device)
        self._second_moment = pl_module.w2_avg.to(self._device)
        self._Dhat = pl_module.Dhat.to(self._device)

    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule") -> bool:
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in pl_module.modules())

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if len(trainer.optimizers) != 1:
            LOGGER.error("SWAG currently works with 1 optimizer.")
            raise RuntimeError

        if len(trainer.lr_scheduler_configs) > 1:
            LOGGER.error("SWAG currently not supported for more than 1 lr_scheduler.")
            raise RuntimeError

        assert trainer.max_epochs is not None, "max_epochs == None in the trainer!"

        has_batch_norm = self.pl_module_contains_batch_norm(pl_module)
        assert not has_batch_norm, "Models with BatchNorm layers are not supported yet!"

        self._max_epochs = trainer.max_epochs

        if self._scheduler_state is not None:
            self._clear_schedulers(trainer)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (not self._initialized) and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            self._initialized = True

            optimizer = trainer.optimizers[0]
            if isinstance(self._swa_lrs, float):
                self._swa_lrs = [self._swa_lrs] * len(optimizer.param_groups)

            for lr, group in zip(self._swa_lrs, optimizer.param_groups):
                group["initial_lr"] = lr

            assert trainer.max_epochs is not None
            self._swa_scheduler = cast(
                LRScheduler,
                SWALR(
                    optimizer,
                    swa_lr=self._swa_lrs,  # type: ignore[arg-type]
                    anneal_epochs=self._annealing_epochs,
                    anneal_strategy=self._annealing_strategy,
                    last_epoch=trainer.max_epochs if self._annealing_strategy == "cos" else -1,
                ),
            )
            if self._scheduler_state is not None:
                # Restore scheduler state from checkpoint
                self._swa_scheduler.load_state_dict(self._scheduler_state)
            elif trainer.current_epoch != self.swa_start:
                # Log a warning if we're initializing after start without any checkpoint data,
                # as behaviour will be different compared to having checkpoint data.
                LOGGER.warning("SWAG is initializing after swa_start without any checkpoint data.")

            # We assert that there is only one optimizer on fit start
            default_scheduler_cfg = LRSchedulerConfig(self._swa_scheduler)
            assert default_scheduler_cfg.interval == "epoch"
            assert default_scheduler_cfg.frequency == 1

            if trainer.lr_scheduler_configs:
                scheduler_cfg = trainer.lr_scheduler_configs[0]
                if scheduler_cfg.interval != "epoch" or scheduler_cfg.frequency != 1:
                    LOGGER.warning("SWA is currently only supported every epoch. Found %s", scheduler_cfg)
                LOGGER.info(
                    "Swapping scheduler `%s` for `%s`",
                    scheduler_cfg.scheduler.__class__.__name__,
                    self._swa_scheduler.__class__.__name__,
                )
                trainer.lr_scheduler_configs[0] = default_scheduler_cfg
            else:
                trainer.lr_scheduler_configs.append(default_scheduler_cfg)

            if self.n_averaged is None:
                self.n_averaged = torch.tensor(self._init_n_averaged, dtype=torch.long, device=pl_module.device)

        if (self.swa_start <= trainer.current_epoch <= self.swa_end) and (trainer.current_epoch > self._latest_update_epoch):
            assert self.n_averaged is not None
            # assert self._swag_model is not None
            self.update_swag_parameters(pl_module)
            self._latest_update_epoch = trainer.current_epoch

    def update_swag_parameters(self, model: pl.LightningModule) -> None:
        r"""
        Updates the SWAG state parameters: first & second moments and the \hat{D} matrix.
        Refer to algorithm 1 in https://arxiv.org/pdf/1902.02476.pdf
        """
        LOGGER.debug("Updating SWAG tensors -- this is update no %d", self.n_averaged)

        flat_w = self.flatten(model.parameters())
        flat_w2 = torch.square(flat_w)

        if self._first_moment is None:
            # initialization
            self._first_moment = flat_w
            self._second_moment = torch.square(flat_w)
        else:
            # update moments
            self._first_moment = self._first_moment + (flat_w - self._first_moment) / (self.n_averaged + 1)
            self._second_moment = self._second_moment + (flat_w2 - self._second_moment) / (self.n_averaged + 1)

        if self._Dhat is None:
            # init the \hat{D} matrix
            self._Dhat = flat_w.clone()[:, None]
        else:
            # store the weights in the \hat{D} matrix - we measure their discrepancy wrt the average later
            self._Dhat = torch.cat((self._Dhat, flat_w[:, None]), dim=1)
            if self._Dhat.shape[1] > self.k:
                # discard the first ("oldest") column of the \hat{D} matrix
                self._Dhat = self._Dhat[:, 1:]

        self.n_averaged += 1

    def flatten(self, params: Iterator[Tuple[str, nn.Parameter]]) -> torch.Tensor:
        """Flattens the model parameters - returns a one-dimensional tensor"""
        p_flat: Optional[torch.Tensor] = None
        for p in params:
            if p_flat is None:
                p_flat = p.detach().clone().to(self._device).reshape(-1)
            else:
                p_flat = torch.cat([p_flat, p.reshape(-1)])
        return p_flat

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n_averaged": 0 if self.n_averaged is None else self.n_averaged.item(),
            "latest_update_epoch": self._latest_update_epoch,
            "scheduler_state": None if self._swa_scheduler is None else self._swa_scheduler.state_dict(),
            "swag_w_avg": None if self._first_moment is None else self._first_moment,
            "swag_w2_avg": None if self._second_moment is None else self._second_moment,
            "Dhat": None if self._Dhat is None else self._Dhat,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._init_n_averaged = state_dict["n_averaged"]
        self._latest_update_epoch = state_dict["latest_update_epoch"]
        self._scheduler_state = state_dict["scheduler_state"]
        self._first_moment(state_dict["swag_w_avg"])
        self._second_moment(state_dict["swag_w2_avg"])
        self._Dhat(state_dict["Dhat"])

    @staticmethod
    def _clear_schedulers(trainer: "pl.Trainer") -> None:
        # If we have scheduler state saved, clear the scheduler configs so that we don't try to
        # load state into the wrong type of schedulers when restoring scheduler checkpoint state.
        # We'll configure the scheduler and re-load its state in on_train_epoch_start.
        if trainer.lr_scheduler_configs:
            assert len(trainer.lr_scheduler_configs) == 1
            trainer.lr_scheduler_configs.clear()
