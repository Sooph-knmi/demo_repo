import os
from typing import Dict, Mapping, Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from timm.scheduler import CosineLRScheduler
import torch
import wandb

from aifs.data.era_normalizers import InputNormalizer
from aifs.losses.wmse import WeightedMSELoss
from aifs.losses.kcrps import KernelCRPS
from aifs.model.msg import GraphMSG
from aifs.utils.logger import get_logger
from aifs.utils.plots import init_plot_settings, plot_predicted_ensemble, plot_kcrps
from aifs.train.utils import pl_scaling
from aifs.metrics.ranks import RankHistogram

LOGGER = get_logger(__name__)


class GraphForecaster(pl.LightningModule):
    def __init__(
        self,
        metadata: Dict,
        config: DictConfig,
    ) -> None:
        super().__init__()

        self.fcdim = config.data.num_features - config.data.num_aux_features
        num_levels = len(config.data.pl.levels)

        self.graph_data = torch.load(os.path.join(config.paths.graph, config.files.graph))

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
        self.metric = WeightedMSELoss(area_weights=self.era_weights)

        self.ensemble_size = config.training.ensemble_size
        # Rank histogram
        if config:
            self.ranks = RankHistogram(nens=self.ensemble_size, nvar=self.fcdim)

        self.multi_step = config.training.multistep_input
        self.lr = config.hardware.num_nodes * config.hardware.num_gpus * config.training.lr.rate
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.log_to_wandb = config.diagnostics.logging.wandb
        self.plot_frequency = config.diagnostics.plot.frequency
        self.save_basedir = os.path.join(config.paths.plots, config.paths.run_id)

        init_plot_settings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def calculate_kcrps(self, y_pred: torch.Tensor, y_target: torch.Tensor, reduce_sum: bool = True) -> torch.Tensor:
        """Rearranges the prediction and ground truth tensors and then computes the KCRPS loss"""
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
        self, batch: torch.Tensor, batch_idx: int, compute_metrics: bool = False, plot: bool = False
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.normalizer(batch)  # normalized in-place
        metrics = {}

        # start rollout
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multistep, latlon, nvar)
        x = torch.stack([x] * self.ensemble_size, dim=1)  # shape == (bs, nens, multistep, latlon, nvar)

        for rstep in range(self.rollout):
            y_pred = self(x)  # prediction at rollout step rstep, shape = (bs, nens, latlon, nvar)
            y = batch[:, self.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)
            loss += self.calculate_kcrps(y_pred, y[..., : self.fcdim])

            if plot and self.global_rank == 0:
                self.plot_sample(batch_idx, rstep, y[..., : self.fcdim], y_pred)
                pointwise_kcrps = self.calculate_kcrps(y_pred, y[..., : self.fcdim], reduce_sum=False)  # shape (nvar, latlon)
                self.plot_pointwise_kcrps(batch_idx, rstep, pointwise_kcrps)

            x = self.advance_input(x, y, y_pred)

            if compute_metrics:
                # rank histograms - update metric state
                _ = self.ranks(y[..., : self.fcdim], y_pred)
                # WMSE metrics
                for mkey, (low, high) in self.metric_ranges.items():
                    y_denorm = self.normalizer.denormalize(y.clone())
                    y_hat_denorm = self.normalizer.denormalize(x[:, :, -1, ...].mean(dim=1).clone())  # ensemble mean
                    metrics[f"{mkey}_{rstep+1}"] = self.metric(y_hat_denorm[..., low:high], y_denorm[..., low:high])

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _ = self._step(batch, batch_idx)
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
        plot = batch_idx % self.plot_frequency == 3
        with torch.no_grad():
            val_loss, metrics = self._step(batch, batch_idx, compute_metrics=True, plot=plot)
        self.log(
            "val_kcrps",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.shape[0],
            rank_zero_only=True,
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
                rank_zero_only=True,
            )

    def plot_pointwise_kcrps(self, batch_idx: int, rollout_step: int, pkcrps: torch.Tensor) -> None:
        """pkcrps.shape == (nvar, latlon)"""
        fig = plot_kcrps(np.rad2deg(self.era_latlons.numpy()), pkcrps.cpu().numpy())
        fig.tight_layout()
        self.output_figure(
            fig,
            tag=f"gnn_val_ens_kcrps_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank{self.global_rank:02d}",
            exp_log_tag=f"val_ens_kcrps_rstep{rollout_step:02d}_rank{self.global_rank:02d}",
        )

    def plot_sample(self, batch_idx: int, rollout_step: int, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """Plots a denormalized sample: input, target and prediction."""
        sample_idx = 0

        y_true_ = self.normalizer.denormalize(y_true.clone()).cpu().numpy()
        y_pred_ = self.normalizer.denormalize(y_pred.clone()).cpu().numpy()

        fig = plot_predicted_ensemble(
            np.rad2deg(self.era_latlons.numpy()),
            y_true_[sample_idx, ...].squeeze(),
            y_pred_[sample_idx, ...].squeeze(),
        )

        fig.tight_layout()
        self.output_figure(
            fig,
            tag=f"gnn_val_ens_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank{self.global_rank:02d}",
            exp_log_tag=f"val_ens_sample_rstep{rollout_step:02d}_rank{self.global_rank:02d}",
        )

    def output_figure(self, fig, tag: str = "gnn", exp_log_tag: str = "val_pred_sample") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = os.path.join(self.save_basedir, f"plots/{tag}_epoch{self.current_epoch:03d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=100)
            if self.log_to_wandb:
                self.logger.experiment.log({exp_log_tag: wandb.Image(save_path)})
        plt.close(fig)  # cleanup

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.95), lr=self.lr)  # , fused=True)
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
