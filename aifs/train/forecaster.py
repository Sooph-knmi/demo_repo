import math
import os
from pathlib import Path
from typing import Dict
from typing import Mapping
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from aifs.data.scaling import pressure_level
from aifs.model.losses import grad_scaler
from aifs.model.losses import WeightedMSELoss
from aifs.model.model import AIFSModelGNN
from aifs.utils.config import DotConfig
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        metadata: Dict,
        config: DictConfig,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        metadata : Dict
            Zarr metadata
        config : DictConfig
            Job configuration
        """
        super().__init__()

        LOGGER.setLevel(config.diagnostics.log.code.level)

        self.fcdim = config.data.num_features - config.data.num_aux_features
        num_levels = len(config.data.pl.levels)

        self.graph_data = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph))

        self.model = AIFSModelGNN(
            metadata=metadata,
            graph_data=self.graph_data,
            config=DotConfig(OmegaConf.to_container(config, resolve=True)),
        )

        self.save_hyperparameters()

        self.era_latlons = self.graph_data[("era", "to", "era")].ecoords_rad
        self.era_weights = self.graph_data[("era", "to", "era")].area_weights

        self.logger_enabled = config.diagnostics.log.wandb.enabled

        loss_scaling = np.array([], dtype=np.float32)
        for pl_name in config.data.pl.parameters:
            if pl_name in config.training.loss_scaling.pl:
                scl = config.training.loss_scaling.pl[pl_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", pl_name)
            loss_scaling = np.append(loss_scaling, [scl] * pressure_level(config.data.pl.levels))
        for sfc_name in config.data.sfc.parameters:
            if sfc_name in config.training.loss_scaling.sfc:
                scl = config.training.loss_scaling.sfc[sfc_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", sfc_name)
            loss_scaling = np.append(loss_scaling, [scl])
        assert len(loss_scaling) == self.fcdim
        loss_scaling = torch.from_numpy(loss_scaling)

        self.loss = WeightedMSELoss(area_weights=self.era_weights, data_variances=loss_scaling)
        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.metric_ranges = {}
        for i, key in enumerate(config.data.pl.parameters):
            self.metric_ranges[key] = [i * num_levels, (i + 1) * num_levels]
        for key in config.training.metrics:
            idx = metadata["name_to_index"][key]
            self.metric_ranges[key] = [idx, idx + 1]
        self.metrics = WeightedMSELoss(area_weights=self.era_weights)

        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.use_zero_optimizer = config.training.zero_optimizer

        self.model_comm_group = None

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.enable_plot = config.diagnostics.plot.enabled

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model
        self.model_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)

    def set_model_comm_group(self, model_comm_group) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    def advance_input(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        x = x.roll(-1, dims=1)
        # autoregressive predictions - we re-init the "variable" part of x
        x[:, self.multi_step - 1, :, : self.fcdim] = y_pred
        # get new "constants" needed for time-varying fields
        x[:, self.multi_step - 1, :, self.fcdim :] = y[..., self.fcdim :]
        return x

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.model.normalizer(batch)  # normalized in-place
        metrics = {}

        # start rollout
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)

        y_preds = []
        for rstep in range(self.rollout):
            # if rstep > 0: torch.cuda.empty_cache() # uncomment if rollout fails with OOM
            y_pred = self(x)  # prediction at rollout step rstep, shape = (bs, latlon, nvar)

            y = batch[:, self.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss += checkpoint(self.loss, y_pred, y[..., : self.fcdim], use_reentrant=False)

            x = self.advance_input(x, y, y_pred)

            if validation_mode:
                for mkey, (low, high) in self.metric_ranges.items():
                    y_denorm = self.model.normalizer.denormalize(y, in_place=False)
                    y_hat_denorm = self.model.normalizer.denormalize(x[:, -1, ...], in_place=False)
                    metrics[f"{mkey}_{rstep+1}"] = self.metrics(y_hat_denorm[..., low:high], y_denorm[..., low:high])

                if self.enable_plot:
                    y_preds.append(y_pred)

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            "train_wmse",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self):
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)
        self.log(
            "val_wmse",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss, y_preds

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.normalizer(batch, in_place=False)

        with torch.no_grad():
            x = batch[:, 0 : self.multi_step, ...]
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat, in_place=False)

    def configure_optimizers(self):
        if self.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(), optimizer_class=torch.optim.AdamW, betas=(0.9, 0.95), lr=self.lr
            )
        else:
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.95), lr=self.lr)  # , fused=True)

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
