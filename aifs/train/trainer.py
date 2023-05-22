import os
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch_geometric.data import HeteroData

# from torch.autograd.graph import save_on_cpu

from aifs.data.era_normalizers import InputNormalizer
from aifs.model.losses import WeightedMSELoss
from aifs.model.msg import GraphMSG
from aifs.utils.logger import get_logger
from aifs.utils.plots import init_plot_settings, plot_loss, plot_predicted_multilevel_flat_sample

LOGGER = get_logger(__name__)


class GraphForecaster(pl.LightningModule):
    _VAL_PLOT_FREQ = 750

    def __init__(
        self,
        graph_data: HeteroData,
        metadata: Dict,
        fc_dim: int,
        aux_dim: int,
        num_levels: int,
        encoder_num_layers: int = 4,
        encoder_mapper_num_layers: int = 1,
        encoder_hidden_channels: int = 128,
        encoder_out_channels: int = 128,
        mlp_extra_layers: int = 0,
        activation: str = "SiLU",
        lr: float = 1e-4,
        rollout: int = 1,
        multistep: int = 1,
        save_basedir: Optional[str] = None,
        log_to_wandb: bool = False,
        loss_scaling: Optional[torch.Tensor] = None,
        pl_names: Optional[List] = None,
        metric_names: Optional[List] = None,
    ) -> None:
        super().__init__()

        self.gnn = GraphMSG(
            graph_data=graph_data,
            in_channels=fc_dim,
            aux_in_channels=aux_dim,
            multistep=multistep,
            encoder_num_layers=encoder_num_layers,
            encoder_hidden_channels=encoder_hidden_channels,
            encoder_out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            encoder_mapper_num_layers=encoder_mapper_num_layers,
        )

        self.save_hyperparameters()

        self.normalizer = InputNormalizer(metadata)

        self.era_latlons = graph_data[("era", "to", "era")].ecoords_rad
        self.era_weights = graph_data[("era", "to", "era")].area_weights

        if loss_scaling is None:
            loss_scaling = torch.ones(1, dtype=torch.float32)  # unit weights

        self.loss = WeightedMSELoss(area_weights=self.era_weights, data_variances=loss_scaling)

        # TODO: what if pl_names is None? either guard against that or make it a required arg
        # or, better yet, can we replace `pl_names` with the level names from the input metadata?
        self.metric_ranges = {}
        for i, key in enumerate(pl_names):
            self.metric_ranges[key] = [i * num_levels, (i + 1) * num_levels]
        for key in metric_names:
            idx = metadata["name_to_index"][key]
            self.metric_ranges[key] = [idx, idx + 1]
        self.metrics = WeightedMSELoss(area_weights=self.era_weights)

        self.fcdim = fc_dim
        self.mstep = multistep
        self.lr = lr
        self.rollout = rollout

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Multistep: %d", self.mstep)

        self.log_to_wandb = log_to_wandb
        self.save_basedir = save_basedir

        init_plot_settings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def advance_input(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        x = x.roll(-1, dims=1)
        # autoregressive predictions - we re-init the "variable" part of x
        x[:, self.mstep - 1, :, : self.fcdim] = y_pred
        # get new "constants" needed for time-varying fields
        x[:, self.mstep - 1, :, self.fcdim :] = y[..., self.fcdim :]
        return x

    def _step(
        self, batch: torch.Tensor, batch_idx: int, compute_metrics: bool = False, plot: bool = False
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.normalizer(batch)  # normalized in-place
        metrics = {}

        # start rollout
        x = batch[:, 0 : self.mstep, ...]  # (bs, mstep, latlon, nvar)

        # with save_on_cpu(pin_memory=True):
        for rstep in range(self.rollout):
            y_pred = self(x)  # prediction at rollout step rstep, shape = (bs, latlon, nvar)
            y = batch[:, self.mstep + rstep, ...]  # target, shape = (bs, latlon, nvar)
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss += self.loss(y_pred, y[..., : self.fcdim])

            if plot and self.global_rank == 0:
                self._plot_loss(y_pred, y[..., : self.fcdim], rollout_step=rstep)
                self._plot_sample(batch_idx, rstep, x[:, -1, :, : self.fcdim], y[..., : self.fcdim], y_pred)

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
        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        plot_sample = batch_idx % self._VAL_PLOT_FREQ == 3
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
        self._output_figure(
            fig,
            tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
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
        plt.close(fig)  # cleanup

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.95), lr=self.lr, fused=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180, eta_min=5.0e-7)
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
