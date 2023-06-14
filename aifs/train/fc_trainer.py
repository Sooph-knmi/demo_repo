import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch_geometric.data import HeteroData

# from torch.autograd.graph import save_on_cpu
from timm.scheduler import CosineLRScheduler

from aifs.data.era_normalizers import InputNormalizer
from aifs.model.losses import WeightedMSELoss
from aifs.model.msg import GraphMSG
from aifs.utils.logger import get_logger
from aifs.utils.plots import init_plot_settings, plot_loss, plot_predicted_sample

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
        encoder_out_channels: int = 512,
        mlp_extra_layers: int = 0,
        era_trainable_size: int = 8,
        h_trainable_size: int = 8,
        e2h_trainable_size: int = 8,
        h2e_trainable_size: int = 8,
        h2h_trainable_size: int = 0,
        activation: str = "SiLU",
        lr: float = 1e-4,
        lr_iterations: int = 300000,
        lr_min: float = 1.5e-7,
        rollout: int = 1,
        multistep: int = 1,
        save_basedir: Optional[str] = None,
        log_to_wandb: bool = False,
        loss_scaling: Optional[torch.Tensor] = None,
        pl_names: Optional[List] = None,
        metric_names: Optional[List] = None,
        rollout_epoch_increment: int = 0,
        rollout_max: int = 12,
    ) -> None:
        super().__init__()

        self.forecaster = GraphMSG(
            graph_data=graph_data,
            in_channels=fc_dim,
            aux_in_channels=aux_dim,
            multistep=multistep,
            encoder_num_layers=encoder_num_layers,
            encoder_out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            encoder_mapper_num_layers=encoder_mapper_num_layers,
            era_trainable_size=era_trainable_size,
            h_trainable_size=h_trainable_size,
            e2h_trainable_size=e2h_trainable_size,
            h2e_trainable_size=h2e_trainable_size,
            h2h_trainable_size=h2h_trainable_size,
        )

        self.save_hyperparameters()

        self.normalizer = InputNormalizer(metadata)

        # we drive stick!
        self.automatic_optimization = False

        self.era_latlons = graph_data[("era", "to", "era")].ecoords_rad
        self.era_weights = graph_data[("era", "to", "era")].area_weights

        if loss_scaling is None:
            loss_scaling = torch.ones(1, dtype=torch.float32)  # unit weights

        self.loss = WeightedMSELoss(area_weights=self.era_weights, data_variances=loss_scaling)

        # TODO: extract the level names from the input metadata
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
        self.lr_iterations = lr_iterations
        self.lr_min = lr_min
        self.rollout = rollout
        self.rollout_epoch_increment = rollout_epoch_increment
        self.rollout_max = rollout_max
        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.mstep)

        self.log_to_wandb = log_to_wandb
        self.save_basedir = save_basedir

        init_plot_settings()

    def load_and_freeze_encoder_weights(self, weights_dict: Dict, prefix: str) -> None:
        with torch.no_grad():
            for name, param in self.forecaster.encoder.named_parameters():
                # init from AE state
                param.copy_(weights_dict[f"{prefix}.{name}"])
        for param in self.forecaster.encoder.parameters():
            # freeze (don't optimize further)
            param.requires_grad = False
        self.forecaster.encoder.eval()

    def load_and_freeze_decoder_weights(self, weights_dict: Dict, prefix: str) -> None:
        with torch.no_grad():
            for name, param in self.forecaster.decoder.named_parameters():
                # init from AE state
                param.copy_(weights_dict[f"{prefix}.{name}"])
        for param in self.forecaster.decoder.parameters():
            # freeze (don't optimize further)
            param.requires_grad = False
        self.forecaster.decoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecaster(x)

    def advance_input(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        x = x.roll(-1, dims=1)
        # autoregressive predictions - we re-init the "variable" part of x
        x[:, self.mstep - 1, :, : self.fcdim] = y_pred
        # get new "constants" needed for time-varying fields
        x[:, self.mstep - 1, :, self.fcdim :] = y[..., self.fcdim :]
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        opt = self.optimizers()

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.normalizer(batch)  # normalized in-place
        # start rollout
        x = batch[:, 0 : self.mstep, ...]  # (bs, mstep, latlon, nvar)

        for rstep in range(self.rollout):
            y_pred = self(x)  # prediction at rollout step rstep, shape = (bs, latlon, nvar)
            y = batch[:, self.mstep + rstep, ...]  # target, shape = (bs, latlon, nvar)
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            loss += self.loss(y_pred, y[..., : self.fcdim])
            x = self.advance_input(x, y, y_pred)

        # scale loss
        loss *= 1.0 / self.rollout

        # TODO: add gradient accumulation
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=32.0, gradient_clip_algorithm="value")
        opt.step()

        sched = self.lr_schedulers()
        sched.step()

        self.log(
            "train_wmse",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            self.rollout,
            on_step=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=False,
        )
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        del metric  # not used
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self):
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        with torch.no_grad():
            plot = batch_idx % self._VAL_PLOT_FREQ == 3
            val_loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
            batch = self.normalizer(batch)
            metrics = {}

            # start rollout
            x = batch[:, 0 : self.mstep, ...]

            for rstep in range(self.rollout):
                y_pred = self(x)
                y = batch[:, self.mstep + rstep, ...]
                val_loss += self.loss(y_pred, y[..., : self.fcdim])

                if plot and self.global_rank == 0:
                    self._plot_loss(y_pred, y[..., : self.fcdim], rollout_step=rstep)
                    self._plot_sample(batch_idx, rstep, x[:, -1, :, : self.fcdim], y[..., : self.fcdim], y_pred.float())

                x = self.advance_input(x, y, y_pred)

                for mkey, (low, high) in self.metric_ranges.items():
                    y_denorm = self.normalizer.denormalize(y.clone())
                    y_hat_denorm = self.normalizer.denormalize(x[:, -1, ...].float().clone())
                    metrics[f"{mkey}_{rstep+1}"] = self.metrics(y_hat_denorm[..., low:high], y_denorm[..., low:high])

            # scale loss
            val_loss *= 1.0 / self.rollout

            self.log(
                "val_wmse",
                val_loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
                rank_zero_only=True,
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

        fig = plot_predicted_sample(
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

    def configure_optimizers(self):
        params = list(self.forecaster.processor.parameters()) + list(self.forecaster.decoder.parameters())
        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.99), lr=self.lr, fused=True)
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
