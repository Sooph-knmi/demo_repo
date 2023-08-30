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
from aifs.diagnostics.plots import plot_kcrps
from aifs.diagnostics.plots import plot_loss
from aifs.diagnostics.plots import plot_predicted_ensemble
from aifs.diagnostics.plots import plot_predicted_multilevel_flat_sample
from aifs.diagnostics.plots import plot_rank_histograms
from aifs.diagnostics.plots import plot_spread_skill

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

    def __init__(self, config) -> None:
        """Initialize RolloutEval callback.

        Parameters
        ----------
        config : dict
            Dictionary with configuration settings
        """
        super().__init__()
        LOGGER.debug(
            "Setting up RolloutEval callback with rollout = %d, frequency = %d ...",
            config.diagnostics.eval.rollout,
            config.diagnostics.eval.frequency,
        )
        self.eval_plot_parameters = config.diagnostics.plot.parameters
        self.rollout = config.diagnostics.eval.rollout
        self.frequency = config.diagnostics.eval.frequency

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        metrics = {}

        assert hasattr(
            pl_module, "spread_skill"
        ), "To use this callback, you must define a `spread_skill` attribute of type SpreadSkill in your Forecaster class!"

        # start rollout
        x = batch[:, 0 : pl_module.multi_step, ...]  # (bs, multistep, latlon, nvar)
        x = torch.stack([x] * pl_module.ensemble_size, dim=1)  # shape == (bs, nens, multistep, latlon, nvar)

        assert batch.shape[1] >= self.rollout + pl_module.multi_step, "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            rmse = torch.zeros((self.rollout, len(self.eval_plot_parameters)), dtype=batch.dtype, device=pl_module.device)
            spread = torch.zeros_like(rmse)

            for rstep in range(self.rollout):
                y_pred = pl_module(x)  # prediction at rollout step rstep, shape = (bs, nens, latlon, nvar)
                y = batch[:, pl_module.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)
                # y includes the auxiliary variables, so we must leave those out when computing the loss
                loss += pl_module.calculate_kcrps(y_pred, y[..., : pl_module.fcdim])

                x = pl_module.advance_input(x, y, y_pred)

                # training metrics
                for mkey, (low, high) in pl_module.metric_ranges.items():
                    y_denorm = pl_module.normalizer.denormalize(y, in_place=False)
                    # ensemble mean
                    y_pred_denorm = pl_module.normalizer.denormalize(x[:, :, -1, ...].mean(dim=1), in_place=False)
                    metrics[f"{mkey}_{rstep+1}"] = pl_module.metrics(y_pred_denorm[..., low:high], y_denorm[..., low:high])

                # eval diagnostic metrics
                for midx, (pidx, pname) in enumerate(self.eval_plot_parameters.items()):
                    y_denorm = pl_module.normalizer.denormalize(y, in_place=False)
                    y_pred_denorm = pl_module.normalizer.denormalize(x[:, :, -1, ...], in_place=False)
                    # ensemble mean RMSE
                    rmse[rstep, midx] = torch.sqrt(
                        pl_module.metrics(y_pred_denorm[..., pidx : pidx + 1].mean(dim=1), y_denorm[..., pidx : pidx + 1])
                    )
                    # mean spread (ensemble stdev)
                    spread[rstep, midx] = y_pred_denorm[..., pidx : pidx + 1].std(dim=1).mean()

                    LOGGER.debug("%s mean RMSE at roll step %d: %.3e", pname, rstep, rmse[rstep, midx])
                    LOGGER.debug("%s spread at roll step %d: %.3e", pname, rstep, spread[rstep, midx])

            # update spread-skill metric state
            _ = pl_module.spread_skill(rmse, spread)

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


class RankHistogramPlot(PlotCallback):
    def __init__(self, config):
        super().__init__(config)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert hasattr(
            pl_module, "ranks"
        ), "To use this callback, you must define a `ranks` attribute of type RankHistogram in your Forecaster class!"
        fig = plot_rank_histograms(self.config.diagnostics.plot.parameters, pl_module.ranks.compute().cpu().numpy())
        self._output_figure(trainer, fig, tag="ens_rank_hist", exp_log_tag=f"val_rank_hist_{pl_module.global_rank}")
        pl_module.ranks.reset()


class SpreadSkillPlot(PlotCallback):
    def __init__(self, config):
        super().__init__(config)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert hasattr(
            pl_module, "spread_skill"
        ), "To use this callback, you must define a `spread_skill` attribute of type SpreadSkill in your Forecaster class!"
        rmse, spread = (r.cpu().numpy() for r in pl_module.spread_skill.compute())
        fig = plot_spread_skill(self.config.diagnostics.plot.parameters, (rmse, spread), pl_module.spread_skill.time_step)
        self._output_figure(trainer, fig, tag="ens_spread_skill", exp_log_tag=f"val_spread_skill_{pl_module.global_rank}")
        pl_module.ranks.reset()


class KCRPSPlot(PlotCallback):
    def __init__(self, config) -> None:
        super().__init__(config)

    def _plot(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del batch  # not used
        LOGGER.debug("len(outputs) = %d", len(outputs))
        LOGGER.debug("Output types: %s", [type(o) for o in outputs])
        LOGGER.debug("Parameters to plot: %s", self.config.diagnostics.plot.parameters)

        for rollout_step in range(pl_module.rollout):
            fig = plot_kcrps(
                self.config.diagnostics.plot.parameters,
                np.rad2deg(pl_module.era_latlons.numpy()),
                outputs[2][rollout_step].cpu().numpy(),
            )

            fig.tight_layout()
            self._output_figure(
                trainer,
                fig,
                tag=f"gnn_kcrps_val_rstep{rollout_step:02d}_batch{batch_idx:05d}_rank{pl_module.global_rank:02d}",
                exp_log_tag=f"val_kcrps_rstep{rollout_step:02d}_rank{pl_module.global_rank:02d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)


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
        del batch_idx
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
                tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:05d}_rank0",
                exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)


class PredictedEnsemblePlot(PlotCallback):
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

        LOGGER.debug("Fields to plot: %s", [v for _, v in self.config.diagnostics.plot.parameters.items()])

        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_ensemble(
                self.config.diagnostics.plot.parameters,
                np.rad2deg(pl_module.era_latlons.numpy()),
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
                tag=f"gnn_pred_ens_val_rstep{rollout_step:02d}_batch{batch_idx:05d}_rank{pl_module.global_rank:02d}",
                exp_log_tag=f"val_pred_ens_rstep{rollout_step:02d}_rank{pl_module.global_rank:02d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)
