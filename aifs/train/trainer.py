import os
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import HeteroData

import wandb
from aifs.model.losses import WeightedMSELoss
from aifs.data.era_normalizers import InputNormalizer
from aifs.model.msg import GraphMSG
from aifs.utils.logger import get_logger
from aifs.utils.plots import plot_predicted_multilevel_flat_sample, init_plot_settings, plot_loss

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
        save_basedir: Optional[str] = None,
        act_checkpoints: bool = True,
        log_to_wandb: bool = False,
        log_to_neptune: bool = False,
        log_persistence: bool = False,
        loss_scaling: Optional[torch.Tensor] = None,
        pl_names: Optional[List] = None,
        metric_names: Optional[List] = None,
        rollout_epoch_increment: int = 0,
        rollout_max: int = 12,
    ) -> None:
        super().__init__()

        self.gnn = GraphMSG(
            graph_data=graph_data,
            in_channels=fc_dim,
            aux_in_channels=aux_dim,
            encoder_num_layers=encoder_num_layers,
            encoder_hidden_channels=encoder_hidden_channels,
            encoder_out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            encoder_mapper_num_layers=encoder_mapper_num_layers,
            act_checkpoints=act_checkpoints,
        )

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

        self.feature_dim = fc_dim
        self.lr = lr
        self.rollout = rollout
        self.rollout_epoch_increment = rollout_epoch_increment
        self.rollout_max = rollout_max
        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)

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
        batch = self.normalizer(batch)  # normalized in-place
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
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=True,
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

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        # manually warm up lr without a scheduler
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        elif self.trainer.global_step < 300000 + 1000:
            lr_scale = ( self.lr - 0.3 * 10**-7 ) * 0.5 * ( 1 + np.cos(np.pi * (self.trainer.global_step-1000)/300000) )
            for pg in optimizer.param_groups:
                pg["lr"] = 0.3 * 10**-7 + lr_scale * self.lr
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = 0.3 * 10**-7
        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_end(self):
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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
        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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
        batch = self.normalizer(batch)

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

    def _shared_eval_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        plot_sample = batch_idx % self._VAL_PLOT_FREQ == 3
        batch = self.normalizer(batch)
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
                if plot_sample and self.global_rank == 0:
                    self._plot_loss(y_hat, y[..., : self.feature_dim], rollout_step=rstep)
                    self._plot_sample(batch_idx, rstep, x[..., : self.feature_dim], y[..., : self.feature_dim], y_hat)
                x[..., : self.feature_dim] = y_hat
                # get new "constants" needed for time-varying fields
                x[..., self.feature_dim :] = y[..., self.feature_dim :]
                for mkey, mranges in self.metric_ranges.items():
                    y_denorm = self.normalizer.denormalize(y.clone())
                    y_hat_denorm = self.normalizer.denormalize(x.clone())
                    low, high = mranges
                    metrics[f"{mkey}_{rstep+1}"] = self.metrics(y_hat_denorm[..., low:high], y_denorm[..., low:high])
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
            if self.log_to_neptune:
                self.logger.experiment[f"val/{tag}_epoch{self.current_epoch:03d}"].upload(save_path)
        plt.close(fig)  # cleanup

    def configure_optimizers(self):
        # TODO: revisit the choice of optimizer (switch to something fancier, like FusedAdam/LAMB?)
        # TODO: Using a momentum-free optimizer (SGD) may reduce memory usage (but degrade convergence?) - to test
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.95), lr=self.lr, fused=True)
#        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180, eta_min=5.0e-7)
        return {
            "optimizer": optimizer,
#            "lr_scheduler": {
#                "scheduler": lr_scheduler,
#                "monitor": "val_wmse",
#                "interval": "epoch",
#                "frequency": 1,
#                "strict": True,
#                "name": "gnn_lr_sched",
#            },
        }
