import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch_geometric.data import HeteroData
from timm.scheduler import CosineLRScheduler

from aifs.data.era_normalizers import InputNormalizer
from aifs.model.losses import WeightedMSELoss
from aifs.model.ae import GraphAE
from aifs.utils.logger import get_logger
from aifs.utils.plots import init_plot_settings, plot_loss, plot_reconstructed_sample

LOGGER = get_logger(__name__)


class GraphAutoencoder(pl.LightningModule):
    _VAL_PLOT_FREQ = 750

    def __init__(
        self,
        graph_data: HeteroData,
        metadata: Dict,
        fc_dim: int,
        aux_dim: int,
        num_levels: int,
        mapper_num_layers: int = 1,
        encoder_out_channels: int = 512,
        mlp_extra_layers: int = 0,
        era_trainable_size: int = 0,
        e2h_trainable_size: int = 0,
        h2e_trainable_size: int = 0,
        activation: str = "SiLU",
        lr: float = 1e-4,
        lr_iterations: int = 50000,
        lr_min: float = 3e-7,
        multistep: int = 1,
        save_basedir: Optional[str] = None,
        log_to_wandb: bool = False,
        loss_scaling: Optional[torch.Tensor] = None,
        pl_names: Optional[List] = None,
        metric_names: Optional[List] = None,
    ) -> None:
        super().__init__()

        self.ae = GraphAE(
            graph_data=graph_data,
            in_channels=fc_dim,
            aux_in_channels=aux_dim,
            encoder_out_channels=encoder_out_channels,
            mlp_extra_layers=mlp_extra_layers,
            activation=activation,
            mapper_num_layers=mapper_num_layers,
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
        self.lr = lr

        self.mstep = multistep
        self.lr_iterations = lr_iterations
        self.lr_min = lr_min

        self.log_to_wandb = log_to_wandb
        self.save_basedir = save_basedir

        init_plot_settings()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decoder(z)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.normalizer(batch)  # normalized in-place

        x = batch[:, 0, ...]  # (bs, latlon, nvar)
        x_hat = self(x)  # reconstructed state
        train_ae_loss = self.loss(x_hat, x[..., : self.fcdim])

        self.log(
            "train_ae_wmse",
            train_ae_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.shape[0],
            sync_dist=True,
        )

        return train_ae_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            batch = self.normalizer(batch)  # normalized in-place

            x = batch[:, 0, ...]  # (bs, latlon, nvar)
            x_hat = self(x)  # reconstructed state
            val_ae_loss = self.loss(x_hat, x[..., : self.fcdim])

            plot = batch_idx % self._VAL_PLOT_FREQ == 3
            if plot and self.global_rank == 0:
                self._plot_ae_loss(x_hat, x[..., : self.fcdim])
                self._plot_reconstructed_sample(batch_idx, x[..., : self.fcdim], x_hat.float())

            metrics = {}
            for mkey, (low, high) in self.metric_ranges.items():
                x_denorm = self.normalizer.denormalize(x.clone())
                x_hat_denorm = self.normalizer.denormalize(x_hat.float().clone())
                metrics[f"{mkey}"] = self.metrics(x_hat_denorm[..., low:high], x_denorm[..., low:high])

            self.log(
                "val_ae_wmse",
                val_ae_loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=True,
                batch_size=batch.shape[0],
                sync_dist=True,
            )

            for mname, mvalue in metrics.items():
                self.log(
                    "val_ae_" + mname,
                    mvalue,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=True,
                    batch_size=batch.shape[0],
                    sync_dist=True,
                )

    def _plot_ae_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        loss = self.loss(y_true, y_pred, squash=False).cpu().numpy()
        fig = plot_loss(loss)
        fig.tight_layout()
        self._output_figure(
            fig,
            tag=f"ae_loss_rank{self.global_rank:01d}",
            exp_log_tag=f"ae_loss_sample_rank{self.global_rank:01d}",
        )

    def _plot_reconstructed_sample(self, batch_idx: int, x_true: torch.Tensor, x_rec: torch.Tensor) -> None:
        """Plots a denormalized sample: input, target and prediction."""
        sample_idx = 0
        x_true_ = self.normalizer.denormalize(x_true.clone()).cpu().numpy()
        x_rec_ = self.normalizer.denormalize(x_rec.clone()).cpu().numpy()

        fig = plot_reconstructed_sample(
            np.rad2deg(self.era_latlons.numpy()),
            x_true_[sample_idx, ...].squeeze(),
            x_rec_[sample_idx, ...].squeeze(),
        )
        fig.tight_layout()
        self._output_figure(
            fig,
            tag=f"gnn_ae_sample_batch{batch_idx:04d}_rank{self.global_rank:01d}",
            exp_log_tag=f"val_ae_sample_rank{self.global_rank:01d}",
        )

    def lr_scheduler_step(self, scheduler, metric):
        del metric  # not used
        scheduler.step(epoch=self.trainer.global_step)

    def _output_figure(self, fig, tag: str = "gnn-ae", exp_log_tag: str = "val_ae_sample") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = os.path.join(self.save_basedir, f"plots/{tag}_epoch{self.current_epoch:03d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=100)
            if self.log_to_wandb:
                self.logger.experiment.log({exp_log_tag: wandb.Image(save_path)})
        plt.close(fig)  # cleanup

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.99), lr=self.lr, fused=False)
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
