import os
from typing import Dict
from typing import Mapping
from typing import Tuple

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from timm.scheduler import CosineLRScheduler

from aifs.data.era_normalizers import InputNormalizer
from aifs.diagnostics.logger import get_logger
from aifs.losses.kcrps import KernelCRPS
from aifs.losses.wmse import WeightedMSELoss
from aifs.metrics.ranks import RankHistogram
from aifs.model.msg import GraphMSG
from aifs.train.utils import pl_scaling

# from aifs.model import grad_scaler

# from torch.autograd.graph import save_on_cpu

LOGGER = get_logger(__name__)


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

        self.fcdim = config.data.num_features - config.data.num_aux_features
        num_levels = len(config.data.pl.levels)

        self.graph_data = torch.load(os.path.join(config.hardware.paths.graph, config.hardware.files.graph))

        self.gnn = GraphMSG(
            config=config,
            graph_data=self.graph_data,
        )

        self.save_hyperparameters()

        self.normalizer = InputNormalizer(metadata)

        self.era_latlons = self.graph_data[("era", "to", "era")].ecoords_rad
        self.era_weights = self.graph_data[("era", "to", "era")].area_weights

        loss_scaling = np.array([], dtype=np.float32)
        for pl_name in config.data.pl.parameters:
            if pl_name in config.training.loss_scaling.pl:
                scl = config.training.loss_scaling.pl[pl_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", pl_name)
            loss_scaling = np.append(loss_scaling, [scl] * pl_scaling(config.data.pl.levels))
        for sfc_name in config.data.sfc.parameters:
            if sfc_name in config.training.loss_scaling.sfc:
                scl = config.training.loss_scaling.sfc[sfc_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", sfc_name)
            loss_scaling = np.append(loss_scaling, [scl])
        assert len(loss_scaling) == self.fcdim
        loss_scaling = torch.from_numpy(loss_scaling)

        # Loss function
        self.kcrps = KernelCRPS(area_weights=self.era_weights, loss_scaling=loss_scaling)

        self.metric_ranges = {}
        for i, key in enumerate(config.data.pl.parameters):
            self.metric_ranges[key] = [i * num_levels, (i + 1) * num_levels]
        for key in config.training.metrics:
            idx = metadata["name_to_index"][key]
            self.metric_ranges[key] = [idx, idx + 1]

        # Validation metric(s)
        self.metrics = WeightedMSELoss(area_weights=self.era_weights)

        self.ensemble_size = config.training.ensemble_size
        # Rank histogram
        if config:
            self.ranks = RankHistogram(nens=self.ensemble_size, nvar=self.fcdim)

        self.multi_step = config.training.multistep_input
        self.lr = config.hardware.num_nodes * config.hardware.num_gpus_per_node * config.training.lr.rate
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.enable_plot = config.diagnostics.plot.enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def calculate_kcrps(self, y_pred: torch.Tensor, y_target: torch.Tensor, reduce_sum: bool = True) -> torch.Tensor:
        """Rearranges the prediction and ground truth tensors and then computes the
        KCRPS loss."""
        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs v latlon e", e=self.ensemble_size)
        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        return self.kcrps(y_pred, y_target, reduce_sum=reduce_sum)

    def advance_input(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # left-shift along the step dimension
        x = x.roll(-1, dims=2)
        # autoregressive predictions - we re-init the "variable" part of x
        x[:, :, self.multi_step - 1, :, : self.fcdim] = y_pred
        # get new "constants" needed for time-varying fields
        x[:, :, self.multi_step - 1, :, self.fcdim :] = y[:, None, :, self.fcdim :]  # add dummy ensemble dim to match x
        return x

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.normalizer(batch)  # normalized in-place
        metrics = {}

        # start rollout
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multistep, latlon, nvar)
        x = torch.stack([x] * self.ensemble_size, dim=1)  # shape == (bs, nens, multistep, latlon, nvar)

        y_preds = []
        for rstep in range(self.rollout):
            y_pred = self(x)  # prediction at rollout step rstep, shape = (bs, nens, latlon, nvar)
            y = batch[:, self.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)
            loss += self.calculate_kcrps(y_pred, y[..., : self.fcdim])

            # THIS NOW HAPPENS IN A CALLBACK
            # if plot and self.global_rank == 0:
            #     self.plot_sample(batch_idx, rstep, y[..., : self.fcdim], y_pred)
            #     pointwise_kcrps = self.calculate_kcrps(y_pred, y[..., : self.fcdim], reduce_sum=False)  # shape (nvar, latlon)
            #     self.plot_pointwise_kcrps(batch_idx, rstep, pointwise_kcrps)

            x = self.advance_input(x, y, y_pred)

            if validation_mode:
                # rank histograms - update metric state
                _ = self.ranks(y[..., : self.fcdim], y_pred)
                # WMSE ensemble mean metrics
                for mkey, (low, high) in self.metric_ranges.items():
                    y_denorm = self.normalizer.denormalize(y, in_place=False)
                    y_hat_denorm = self.normalizer.denormalize(x[:, :, -1, ...].mean(dim=1), in_place=False)  # ensemble mean
                    metrics[f"{mkey}_{rstep+1}"] = self.metric(y_hat_denorm[..., low:high], y_denorm[..., low:high])

                if self.enable_plot:
                    y_preds.append(y_pred)

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        self.log(
            "train_kcrps",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=True,
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
            "val_kcrps",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
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
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss, y_preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.95), lr=self.lr)  # , fused=True)
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
