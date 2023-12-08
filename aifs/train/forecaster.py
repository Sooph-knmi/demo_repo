import math
import os
from collections import defaultdict
from pathlib import Path
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
from aifs.utils.jsonify import map_config_to_primitives
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        statistics : dict
            Statistics of the training data
        data_indices : dict
            Indices of the training data,
        metadata : dict
            Provenance information
        """
        super().__init__()

        LOGGER.setLevel(config.diagnostics.log.code.level)

        self.graph_data = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph), map_location=self.device)

        self.model = AIFSModelGNN(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=self.graph_data,
            config=DotConfig(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        self.data_indices = data_indices

        self.save_hyperparameters()

        self.data_latlons = self.graph_data[("era", "to", "era")].ecoords_rad
        self.area_weights = self.graph_data[("era", "to", "era")].area_weights

        self.logger_enabled = config.diagnostics.log.wandb.enabled

        self.metric_ranges, loss_scaling = self.metrics_loss_scaling(config, data_indices)
        self.loss = WeightedMSELoss(area_weights=self.area_weights, data_variances=loss_scaling)
        self.metrics = WeightedMSELoss(area_weights=self.area_weights)

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

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

    @staticmethod
    def metrics_loss_scaling(config: DictConfig, data_indices):
        metric_ranges = defaultdict(list)
        loss_scaling = np.ones((len(data_indices.data.output.full),), dtype=np.float32) * config.training.loss_scaling.default
        for key, idx in data_indices.model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1:
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges[f"pl_{split[0]}"].append(idx)
                # Create pressure levels in loss scaling vector
                if split[0] in config.training.loss_scaling.pl:
                    loss_scaling[idx] = config.training.loss_scaling.pl[split[0]] * pressure_level(int(split[1]))
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                metric_ranges[f"sfc_{key}"].append(idx)
                # Create surface variables in loss scaling vector
                if key in config.training.loss_scaling.sfc:
                    loss_scaling[idx] = config.training.loss_scaling.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges[key] = [idx]
        loss_scaling = torch.from_numpy(loss_scaling)
        return metric_ranges, loss_scaling

    def set_model_comm_group(self, model_comm_group) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    def advance_input(self, batch: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        x = batch[:, 0 : self.multi_step, ...].roll(-1, dims=1)

        x[:, self.multi_step - 1, :, self.data_indices.model.input.prognostic] = y_pred[
            ..., self.data_indices.model.output.prognostic
        ]
        # get new "constants" needed for time-varying fields
        x[:, self.multi_step - 1, :, self.data_indices.model.input.forcing] = batch[
            :, self.multi_step, ..., self.data_indices.data.input.forcing
        ]
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
            y_pred = self(x[..., self.data_indices.data.input.full])  # prediction at rollout step rstep, shape = (bs, latlon, nvar)

            y = batch[:, self.multi_step + rstep, ..., self.data_indices.data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)

            x = self.advance_input(batch, y_pred)

            if validation_mode:
                y_denorm = self.model.normalizer.denormalize(y, in_place=False)
                y_pred_denorm = self.model.normalizer.denormalize(y_pred, in_place=False)
                for mkey, indices in self.metric_ranges.items():
                    metrics[f"{mkey}_{rstep+1}"] = self.metrics(y_pred_denorm[..., indices], y_denorm[..., indices])

                if self.enable_plot:
                    y_preds.append(y_pred.detach())

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
