import os
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

import wandb
from aifs.diagnostics.logger import get_logger
from aifs.diagnostics.plots import init_plot_settings
from aifs.diagnostics.plots import plot_graph_features
from aifs.diagnostics.plots import plot_loss
from aifs.diagnostics.plots import plot_predicted_multilevel_flat_sample

LOGGER = get_logger(__name__)


class PlotCallback(Callback):
    """Factory for creating a callback that plots data to Weights and Biases."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_basedir = os.path.join(config.hardware.paths.plots, config.hardware.paths.run_id)
        self.plot_frequency = config.diagnostics.plot.frequency
        init_plot_settings()

    def _output_figure(self, trainer, fig, tag: str = "gnn", exp_log_tag: str = "val_pred_sample") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = os.path.join(
                self.save_basedir,
                f"plots/{tag}_epoch{trainer.current_epoch:03d}.png",
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=100)
            if self.config.diagnostics.logging.wandb:
                trainer.logger.experiment.log({exp_log_tag: wandb.Image(fig)})
        plt.close(fig)  # cleanup


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window."""

    def __init__(self, rollout: int = 12, frequency: int = 20) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        rollout : int, optional
            Number of timesteps to roll out, by default 12
        frequency : int, optional
            Frequency of rollout evaluation, in terms of number of batches, by default 20
        """
        super().__init__()
        LOGGER.debug(
            "Setting up RolloutEval callback with rollout = %d, frequency = %d ...",
            rollout,
            frequency,
        )
        self.rollout = rollout
        self.frequency = frequency

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        # NB! the batch is already normalized in-place - see pl_model.validation_step()
        metrics = {}

        # start rollout
        x = batch[:, 0 : pl_module.multi_step, ...]  # (bs, multi_step, latlon, nvar)

        assert batch.shape[1] >= self.rollout + pl_module.multi_step, "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            for rstep in range(self.rollout):
                y_pred = pl_module(x)  # prediction at rollout step rstep, shape = (bs, latlon, nvar)
                y = batch[:, pl_module.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)
                # y includes the auxiliary variables, so we must leave those out when computing the loss
                loss += pl_module.loss(y_pred, y[..., : pl_module.fcdim])

                x = pl_module.advance_input(x, y, y_pred)

                for mkey, (low, high) in pl_module.metric_ranges.items():
                    y_denorm = pl_module.normalizer.denormalize(y, in_place=False)
                    y_pred_denorm = pl_module.normalizer.denormalize(x[:, -1, ...], in_place=False)
                    metrics[f"{mkey}_{rstep+1}"] = pl_module.metrics(y_pred_denorm[..., low:high], y_denorm[..., low:high])

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch.shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: Dict, bs: int) -> None:
        pl_module.log(
            f"val_r{self.rollout}_wmse",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=False,
            rank_zero_only=True,
        )
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"val_r{self.rollout}_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=False,
                rank_zero_only=True,
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        del trainer, outputs  # not used
        if batch_idx % self.frequency == 3 and pl_module.global_rank == 0:
            self._eval(pl_module, batch)


class GraphTrainableFeaturesPlot(PlotCallback):
    """Visualize the trainable features defined at the ERA and H graph nodes, if any.

    TODO: How best to visualize the learned edge embeddings? Offline, perhaps - using code from @Simon's notebook?
    """

    def __init__(self, config):
        super().__init__(config)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.global_rank == 0:
            gnn = pl_module.gnn
            graph = pl_module.graph_data

            ecoords = np.rad2deg(graph[("era", "to", "era")].ecoords_rad.numpy())
            hcoords = np.rad2deg(graph[("h", "to", "h")].hcoords_rad.numpy())

            if gnn.era_trainable is not None:
                fig = plot_graph_features(ecoords, gnn.era_trainable.cpu())
                self._output_figure(trainer, fig, tag="era_trainable", exp_log_tag="era_trainable")

            if gnn.h_trainable is not None:
                fig = plot_graph_features(hcoords, gnn.h_trainable.cpu())
                self._output_figure(trainer, fig, tag="h_trainable", exp_log_tag="h_trainable")


class PlotLoss(PlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(self, config):
        super().__init__(config)

    def _plot(
        # self, y_true: torch.Tensor, y_pred: torch.Tensor, rollout_step: int
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        for rollout_step in range(pl_module.rollout):
            y_hat = outputs[1][rollout_step]
            y_true = batch[:, pl_module.multi_step + rollout_step, :, : pl_module.fcdim]
            loss = pl_module.loss(y_hat, y_true, squash=False).cpu().numpy()

            fig = plot_loss(loss)
            fig.tight_layout()
            self._output_figure(
                trainer,
                fig,
                tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)


class PlotSample(PlotCallback):
    """Plots a denormalized sample: input, target and prediction."""

    def __init__(self, config):
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def _plot(
        # batch_idx: int, rollout_step: int, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor,
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        data = (
            pl_module.normalizer.denormalize(
                batch[self.sample_idx, pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1, ...], in_place=False
            )
            .cpu()
            .numpy()
        )

        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_multilevel_flat_sample(
                self.config.diagnostics.plot.parameters,
                self.config.diagnostics.plot.per_sample,
                np.rad2deg(pl_module.era_latlons.numpy()),
                data[0, ..., : pl_module.fcdim].squeeze(),
                data[rollout_step + 1, ..., : pl_module.fcdim].squeeze(),
                pl_module.normalizer.denormalize(outputs[1][rollout_step][self.sample_idx, ..., : pl_module.fcdim], in_place=False)
                .squeeze()
                .cpu()
                .numpy(),
            )

            fig.tight_layout()
            self._output_figure(
                trainer,
                fig,
                tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)


class PlotSwagSampleCallback(PlotCallback):
    """Plots an ensemble of predictions generated with SWAG.

    See https://arxiv.org/pdf/1902.02476.pdf
    To be used in conjunction with the SWAGForecaster trainer class.
    """

    def __init__(self, config):
        super().__init__(config)
        self.nens = self.config.training.swag.ensemble_size
        self.sample_idx = self.config.diagnostics.plot.sample_idx
        LOGGER.debug("Initialized the SWAG plot callback with nens = %d and sample_index = %d...", self.nens, self.sample_idx)

    def _eval(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        # start rollout
        x = batch[:, 0 : pl_module.multi_step, ...]  # (bs, multi_step, latlon, nvar)
        assert pl_module.rollout == 1, "Rollouts > 1 not yet supported."

        with torch.no_grad():
            y_preds = []
            for _ in range(self.nens):
                y_preds.append(pl_module.forward_swag(x))
            y_preds: torch.Tensor = torch.stack(y_preds, dim=1)  # (bs, nens, latlon, nvar)
            y = batch[:, pl_module.multi_step, :, : pl_module.fcdim]  # target, shape = (bs, latlon, nvar)
            self._plot(y_preds, y, trainer, pl_module, batch_idx)

    def _plot(
        self,
        y_preds,
        y,
        trainer,
        pl_module,
        batch_idx,
    ) -> None:
        pl_module.normalizer.denormalize(y_preds, in_place=False).cpu().numpy()
        pl_module.normalizer.denormalize(y, in_place=False).cpu().numpy()

        # fig = plot_predicted_ensemble(
        #     self.config.diagnostics.plot.parameters,
        #     np.rad2deg(pl_module.era_latlons.numpy()),
        #     predictions[self.sample_idx, ...].squeeze(),
        #     truth[self.sample_idx, ...].squeeze(),
        # )
        # fig.tight_layout()
        # self._output_figure(
        #     trainer,
        #     fig,
        #     tag=f"gnn_pred_swag_ens_rstep00_batch{batch_idx:04d}_rank0",
        #     exp_log_tag=f"gnn_pred_swag_ens_rstep00_batch{batch_idx:04d}_rank0",
        # )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)
