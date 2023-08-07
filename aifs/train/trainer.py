import os
from typing import Dict
from typing import Mapping
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from timm.scheduler import CosineLRScheduler

import wandb
from aifs.data.era_normalizers import InputNormalizer
from aifs.model.losses import grad_scaler
from aifs.model.losses import WeightedMSELoss
from aifs.model.msg import GraphMSG
from aifs.train.utils import pl_scaling
from aifs.utils.logger import get_logger
from aifs.utils.plots import init_plot_settings
from aifs.utils.plots import plot_loss
from aifs.utils.plots import plot_predicted_multilevel_flat_sample

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

        self.log_to_wandb = config.diagnostics.logging.wandb
        self.plot_frequency = config.diagnostics.plot.frequency
        self.save_basedir = os.path.join(config.hardware.paths.plots, config.hardware.paths.run_id)

        init_plot_settings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

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
        compute_metrics: bool = False,
        plot: bool = False,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.normalizer(batch)  # normalized in-place
        metrics = {}

        # start rollout
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)

        # with save_on_cpu(pin_memory=True):
        for rstep in range(self.rollout):
            y_pred = self(x)  # prediction at rollout step rstep, shape = (bs, latlon, nvar)
            y = batch[:, self.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss += self.loss(y_pred, y[..., : self.fcdim])

            if plot and self.global_rank == 0:
                self._plot_loss(y_pred, y[..., : self.fcdim], rollout_step=rstep)
                self._plot_sample(
                    batch_idx,
                    rstep,
                    x[:, -1, :, : self.fcdim],
                    y[..., : self.fcdim],
                    y_pred,
                )

            x = self.advance_input(x, y, y_pred)

            if compute_metrics:
                for mkey, (low, high) in self.metric_ranges.items():
                    y_denorm = self.normalizer.denormalize(y.clone())
                    y_hat_denorm = self.normalizer.denormalize(x[:, -1, ...].clone())
                    metrics[f"{mkey}_{rstep+1}"] = self.metrics(y_hat_denorm[..., low:high], y_denorm[..., low:high])

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        train_loss, _ = self._step(batch, batch_idx)
        self.log(
            "train_wmse",
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
        plot_sample = batch_idx % self.plot_frequency == 3
        with torch.no_grad():
            val_loss, metrics = self._step(batch, batch_idx, compute_metrics=True, plot=plot_sample)
        self.log(
            "val_wmse",
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

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.normalizer(batch)

        with torch.no_grad():
            # start rollout
            x = batch[:, 0 : self.multi_step, ...]
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat)

    def _plot_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, rollout_step: int) -> None:
        loss = self.loss(y_true, y_pred, squash=False).cpu().numpy()
        fig = plot_loss(loss)
        fig.tight_layout()
        self.output_figure(
            fig,
            tag=f"loss_rstep_rstep{rollout_step:02d}_rank{self.local_rank:01d}",
            exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{self.local_rank:01d}",
        )

    def _plot_sample(self, batch_idx: int, rollout_step: int, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """Plots a denormalized sample: input, target and prediction."""
        sample_idx = 0
        x_ = self.normalizer.denormalize(x.clone()).cpu().numpy()
        y_true_ = self.normalizer.denormalize(y_true.clone()).cpu().numpy()
        y_pred_ = self.normalizer.denormalize(y_pred.clone()).cpu().numpy()

        fig = plot_predicted_multilevel_flat_sample(
            np.rad2deg(self.era_latlons.numpy()),
            x_[sample_idx, ...].squeeze(),
            y_true_[sample_idx, ...].squeeze(),
            y_pred_[sample_idx, ...].squeeze(),
        )
        fig.tight_layout()
        self.output_figure(
            fig,
            tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
            exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{self.local_rank:01d}",
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
