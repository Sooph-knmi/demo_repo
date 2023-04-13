import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import HeteroData

import wandb
from gnn_era5.architecture.losses import WeightedMSELoss
from gnn_era5.architecture.msg import GraphMSG
from gnn_era5.data.era_datamodule import ERA5DataBatch
from gnn_era5.utils.logger import get_logger
from gnn_era5.utils.plots import plot_predicted_multilevel_flat_sample, init_plot_settings, plot_loss

LOGGER = get_logger(__name__)


class GraphForecaster(pl.LightningModule):
    _VAL_PLOT_FREQ = 750

    def __init__(
        self,
        graph_data: HeteroData,
        fc_dim: int,
        aux_dim: int,
        num_levels: int,
        encoder_num_layers: int = 4,
        encoder_mapper_num_layers: int = 1,
        encoder_hidden_channels: int = 128,
        encoder_out_channels: int = 128,
        activation: str = "SiLU",
        lr: float = 1e-4,
        rollout: int = 1,
        save_basedir: Optional[str] = None,
        act_checkpoints: bool = True,
        log_to_wandb: bool = False,
        log_to_neptune: bool = False,
        log_persistence: bool = False,
        loss_scaling: Optional[torch.Tensor] = None,
        pl_names: Optional[List] = None,
    ) -> None:
        super().__init__()

        self.gnn = GraphMSG(
            graph_data=graph_data,
            in_channels=fc_dim,
            aux_in_channels=aux_dim,
            encoder_num_layers=encoder_num_layers,
            encoder_hidden_channels=encoder_hidden_channels,
            encoder_out_channels=encoder_out_channels,
            activation=activation,
            encoder_mapper_num_layers=encoder_mapper_num_layers,
            act_checkpoints=act_checkpoints,
        )

        self.era_latlons = graph_data[("era", "to", "era")].ecoords_rad
        self.era_weights = graph_data[("era", "to", "era")].area_weights

        if loss_scaling is None:
            loss_scaling = torch.ones(1, dtype=torch.float32)  # unit weights

        self.loss = WeightedMSELoss(area_weights=self.era_weights, data_variances=loss_scaling)
        self.metric_ranges = {}
        for i, key in enumerate(pl_names):
            self.metric_ranges[key] = [i * num_levels, (i + 1) * num_levels]
        self.metrics = WeightedMSELoss(area_weights=self.era_weights)

        self.feature_dim = fc_dim
        self.lr = lr
        self.rollout = rollout
        LOGGER.debug("Rollout window length: %d", self.rollout)

        self.log_to_wandb = log_to_wandb
        self.log_to_neptune = log_to_neptune
        self.save_basedir = save_basedir

        self.log_persistence = log_persistence

        self.save_hyperparameters()
        init_plot_settings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        train_loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        persist_loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)  # persistence
        # start rollout
        x = batch[:, 0, ...]
        for rstep in range(self.rollout):
            y_hat = self(x)  # prediction at rollout step rstep
            y = batch[:, rstep + 1, ...]  # target
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            train_loss += self.loss(y_hat, y[..., : self.feature_dim])
            persist_loss += self.loss(x[..., : self.feature_dim], y[..., : self.feature_dim])
            # autoregressive predictions - we re-init the "variable" part of x
            x[..., : self.feature_dim] = y_hat
            # get new "constants" needed for time-varying fields
            x[..., self.feature_dim :] = y[..., self.feature_dim :]
        # scale loss
        train_loss *= 1.0 / self.rollout
        persist_loss *= 1.0 / self.rollout

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
        if self.log_persistence:
            self.log(
                "train_persist_wmse",
                persist_loss,
                on_epoch=True,
                on_step=True,
                prog_bar=False,
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
            )

        return train_loss

    def validation_step(self, batch: ERA5DataBatch, batch_idx: int) -> torch.Tensor:
        val_loss, persist_loss, metrics = self._shared_eval_step(batch, batch_idx)
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
        if self.log_persistence:
            self.log(
                "val_persist_wmse",
                persist_loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        for metric in metrics.keys():
            self.log(
                "val_" + metric,
                metrics[metric],
                on_epoch=True,
                on_step=False,
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss

    def test_step(self, batch: ERA5DataBatch, batch_idx: int) -> torch.Tensor:
        test_loss, persist_loss, _ = self._shared_eval_step(batch, batch_idx)
        self.log(
            "test_wmse",
            test_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.shape[0],
            sync_dist=True,
        )

        if self.log_persistence:
            self.log(
                "test_persist_wmse",
                persist_loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return test_loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            # start rollout
            x = batch[:, 0, ...]
            for rstep in range(self.rollout):
                y_hat = self(x)
                x[..., : self.feature_dim] = y_hat
                if rstep + 1 < self.rollout:
                    # get new "constants" needed for time-varying fields
                    x[..., self.feature_dim :] = batch[:, rstep + 1, :, self.feature_dim :]
                preds.append(y_hat)
        return torch.stack(preds, dim=-1)  # stack along new last dimension, return sample indices too

    def _shared_eval_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, ...]:
        plot_sample = batch_idx % self._VAL_PLOT_FREQ == 3
        metrics = {}
        with torch.no_grad():
            loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
            persist_loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)  # persistence loss
            x = batch[:, 0, ...]
            for rstep in range(self.rollout):
                y_hat = self(x)
                y = batch[:, rstep + 1, ...]
                loss += self.loss(y_hat, y[..., : self.feature_dim])
                persist_loss += self.loss(x[..., : self.feature_dim], y[..., : self.feature_dim])
                for metric_key in self.metric_ranges.keys():
                    low, high = self.metric_ranges[metric_key]
                    metrics[f"{metric_key}_{rstep+1}"] = self.metrics(y_hat[..., low:high], y[..., low:high])
                if plot_sample and self.global_rank == 0:
                    self._plot_loss(y_hat, y[..., : self.feature_dim], rollout_step=rstep)
                    self._plot_sample(batch_idx, rstep, x[..., : self.feature_dim], y[..., : self.feature_dim], y_hat)
                x[..., : self.feature_dim] = y_hat
                # get new "constants" needed for time-varying fields
                x[..., self.feature_dim :] = y[..., self.feature_dim :]
            loss *= 1.0 / self.rollout
            persist_loss *= 1.0 / self.rollout
        return loss, persist_loss, metrics

    def _plot_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, rollout_step: int) -> None:
        loss = self.loss(y_true, y_pred, squash=False).cpu().numpy()
        fig = plot_loss(loss)
        fig.tight_layout()
        self._output_figure(
            fig,
            tag=f"loss_rstep_rstep{rollout_step:02d}_rank{self.local_rank:01d}",
            exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{self.local_rank:01d}",
        )

    def _plot_sample(self, batch_idx: int, rollout_step: int, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        sample_idx = 0
        fig = plot_predicted_multilevel_flat_sample(
            np.rad2deg(self.era_latlons.numpy()),
            x[sample_idx, ...].cpu().numpy().squeeze(),
            y_true[sample_idx, ...].cpu().numpy().squeeze(),
            y_pred[sample_idx, ...].cpu().numpy().squeeze(),
        )
        fig.tight_layout()
        self._output_figure(
            fig,
            tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0000",
            exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{self.local_rank:01d}",
        )

    def _output_figure(self, fig, tag: str = "gnn", exp_log_tag: str = "val_pred_sample") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = os.path.join(self.save_basedir, f"plots/{tag}_epoch{self.current_epoch:03d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=100)
            if self.log_to_wandb:
                self.logger.experiment.log({exp_log_tag: wandb.Image(save_path)})
            if self.log_to_neptune:
                self.logger.experiment[f"val/{tag}_epoch{self.current_epoch:03d}"].upload(save_path)
        plt.close(fig)  # cleanup

    def configure_optimizers(self):
        # TODO: revisit the choice of optimizer (switch to something fancier, like FusedAdam/LAMB?)
        # TODO: Using a momentum-free optimizer (SGD) may reduce memory usage (but degrade convergence?) - to test
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.95), lr=self.lr, fused=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300000, eta_min=5.0e-7)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_wmse",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": "gnn_lr_sched",
            },
        }
