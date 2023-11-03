from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from aifs.diagnostics.plots import init_plot_settings
from aifs.diagnostics.plots import plot_ensemble_initial_conditions
from aifs.diagnostics.plots import plot_kcrps
from aifs.diagnostics.plots import plot_loss
from aifs.diagnostics.plots import plot_predicted_ensemble
from aifs.diagnostics.plots import plot_rank_histograms
from aifs.diagnostics.plots import plot_spread_skill
from aifs.distributed.helpers import gather_tensor
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class PlotCallback(Callback):
    """Factory for creating a callback that plots data to Weights and Biases."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_basedir = config.hardware.paths.plots
        self.plot_frequency = config.diagnostics.plot.frequency
        init_plot_settings()

    def _output_figure(self, trainer, fig, tag: str = "gnn", exp_log_tag: str = "val_pred_sample") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}.png",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            if self.config.diagnostics.log.wandb.enabled:
                import wandb

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
        self.loss_type = config.training.loss

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
        ens_ic: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        metrics = {}

        assert hasattr(
            pl_module, "spread_skill"
        ), "To use this callback, you must define a `spread_skill` attribute of type SpreadSkill in your Forecaster class!"

        # start rollout
        x = ens_ic  # shape == (bs, nens, multistep, latlon, nvar), already transformed (?)

        assert batch.shape[1] >= self.rollout + pl_module.multi_step, "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            rmse = torch.zeros((self.rollout, len(self.eval_plot_parameters)), dtype=batch.dtype, device=pl_module.device)
            spread = torch.zeros_like(rmse)

            for rstep in range(self.rollout):
                y_pred = pl_module(x)  # prediction at rollout step rstep, shape = (bs, nens, latlon, nvar)
                y = batch[:, pl_module.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)

                loss_rstep, y_pred_group = pl_module.gather_and_compute_loss(
                    y_pred, y[..., : pl_module.fcdim], validation_mode=True
                )
                loss += loss_rstep
                x = pl_module.advance_input(x, y, y_pred)
                assert y_pred_group is not None

                # training metrics
                for mkey, (low, high) in pl_module.metric_ranges.items():
                    y_denorm = pl_module.model.normalizer.denormalize(y, in_place=False)
                    # ensemble mean
                    y_pred_denorm = pl_module.model.normalizer.denormalize(y_pred_group.mean(dim=1), in_place=False)
                    metrics[f"{mkey}_{rstep+1}"] = pl_module.metrics(y_pred_denorm[..., low:high], y_denorm[..., low:high])

                # eval diagnostic metrics
                for midx, (pidx, _) in enumerate(self.eval_plot_parameters.items()):
                    y_denorm = pl_module.model.normalizer.denormalize(y, in_place=False)
                    y_pred_denorm = pl_module.model.normalizer.denormalize(y_pred_group, in_place=False)
                    # ensemble mean RMSE
                    rmse[rstep, midx] = torch.sqrt(
                        pl_module.metrics(y_pred_denorm[..., pidx : pidx + 1].mean(dim=1), y_denorm[..., pidx : pidx + 1])
                    )
                    # mean spread (ensemble stdev)
                    spread[rstep, midx] = y_pred_denorm[..., pidx : pidx + 1].std(dim=1).mean()

            # update spread-skill metric state
            _ = pl_module.spread_skill(rmse, spread)

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch.shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: Dict, bs: int) -> None:
        pl_module.log(
            f"val_r{self.rollout}_" + self.loss_type,
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=pl_module.logger_enabled,
            batch_size=bs,
            sync_dist=True,
            rank_zero_only=True,
        )
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"val_r{self.rollout}_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=pl_module.logger_enabled,
                batch_size=bs,
                sync_dist=True,
                rank_zero_only=True,
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> None:
        del trainer  # not used
        # this must happen on ALL devices!
        if batch_idx % self.frequency == 3:
            self._eval(pl_module, batch[0], outputs[-1])


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


class KCRPSMapPlot(PlotCallback):
    def __init__(self, config) -> None:
        super().__init__(config)

    def _plot(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del batch  # not used
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


class KCRPSBarPlot(PlotCallback):
    """Plots the unsqueezed loss over rollouts."""

    def __init__(self, config):
        super().__init__(config)
        self.eval_frequency = 10
        assert config.training.loss == "kcrps", "For now, this callback works only with the KCRPS loss ..."

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
            y_true = batch[0][:, pl_module.multi_step + rollout_step, :, : pl_module.fcdim]
            LOGGER.debug("y_hat = %s, y_true = %s", y_hat.shape, y_true.shape)

            kcrps_: torch.Tensor = pl_module._compute_kcrps(y_hat, y_true, squash=False)
            LOGGER.debug("raw kcrps_.shape = %s", kcrps_.shape)
            kcrps_ = kcrps_.sum(dim=-1) / pl_module.kcrps.weights.sum()
            LOGGER.debug("summed kcrps_.shape = %s", kcrps_.shape)

            fig = plot_loss(kcrps_.cpu().numpy())
            fig.tight_layout()
            self._output_figure(
                trainer,
                fig,
                tag=f"loss_rstep{rollout_step:02d}_rank{pl_module.local_rank:02d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:02d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)


# class GraphTrainableFeaturesPlot(AsyncPlotCallback):
#     """Visualize the trainable features defined at the ERA and H graph nodes, if any.

#     TODO: How best to visualize the learned edge embeddings? Offline, perhaps - using code from @Simon's notebook?
#     """

#     def __init__(self, config):
#         super().__init__(config)

#     def _plot(
#         # self, trainer, latlons:np.ndarray, features:np.ndarray, tag:str, exp_log_tag:str
#         self,
#         trainer,
#         latlons,
#         features,
#         epoch,
#         tag,
#         exp_log_tag,
#     ) -> None:
#         fig = plot_graph_features(latlons, features)
#         self._output_figure(trainer, fig, tag=tag, exp_log_tag=exp_log_tag)

#     def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
#         if pl_module.global_rank == 0:
#             model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model
#             graph = pl_module.graph_data
#             epoch = trainer.current_epoch

#             if model.era_trainable is not None:
#                 ecoords = np.rad2deg(graph[("era", "to", "era")].ecoords_rad.numpy())

#                 self._executor.submit(
#                     self._plot,
#                     trainer,
#                     ecoords,
#                     model.era_trainable.cpu(),
#                     epoch=epoch,
#                     tag="era_trainable",
#                     exp_log_tag="era_trainable",
#                 )

#             if model.h_trainable is not None:
#                 hcoords = np.rad2deg(graph[("h", "to", "h")].hcoords_rad.numpy())
#                 self._executor.submit(
#                     self._plot, trainer, hcoords,
#                     model.h_trainable.cpu(), epoch=epoch,
#                     tag="h_trainable", exp_log_tag="h_trainable"
#                 )


class PlotEnsembleInitialConditions(PlotCallback):
    def __init__(self, config):
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def _plot(
        self,
        trainer,
        pl_module,
        ens_ic,
        batch_idx,
    ) -> None:
        denorm_ens_ic = pl_module.model.normalizer.denormalize(ens_ic[self.sample_idx, ...], in_place=False).cpu().numpy()

        for step in range(pl_module.multi_step):
            # plot the perturbations at current step
            fig = plot_ensemble_initial_conditions(
                self.config.diagnostics.plot.parameters,
                np.rad2deg(pl_module.era_latlons.numpy()),
                denorm_ens_ic[:, step, ...].squeeze(),
            )
            fig.tight_layout()
            self._output_figure(
                trainer,
                fig,
                tag=f"gnn_ens_ic_val_mstep{step:02d}_batch{batch_idx:05d}_rank{pl_module.global_rank:02d}",
                exp_log_tag=f"gnn_ens_ic_val_mstep{step:02d}_rank{pl_module.global_rank:02d}",
            )

    def _gather_group_initial_conditions(self, pl_module: pl.LightningModule, my_ens_ic: torch.Tensor) -> torch.Tensor:
        """Gathers all the initial conditions used in a device group to a single
        tensor."""
        group_ens_ic = gather_tensor(
            my_ens_ic, dim=1, shapes=[my_ens_ic.shape] * pl_module.mgroupdef[1], mgroup=pl_module.mgroupdef[0]
        )
        return group_ens_ic

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del batch  # not used
        group_ens_ic = self._gather_group_initial_conditions(pl_module, outputs[-1])
        # plotting happens only on device 0
        # device 0 has already gathered all ensemble members generated by its group
        if batch_idx % self.plot_frequency == 3 and pl_module.global_rank == 0:
            self._plot(trainer, pl_module, group_ens_ic, batch_idx)


class PredictedEnsemblePlot(PlotCallback):
    def __init__(self, config):
        super().__init__(config)
        self.sample_idx = self.config.diagnostics.plot.sample_idx

    def _plot(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        data = (
            pl_module.model.normalizer.denormalize(
                batch[0][self.sample_idx, pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1, ...],
                in_place=False,
            )
            .cpu()
            .numpy()
        )

        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_ensemble(
                self.config.diagnostics.plot.parameters,
                np.rad2deg(pl_module.era_latlons.numpy()),
                data[rollout_step + 1, ..., : pl_module.fcdim].squeeze(),
                pl_module.model.normalizer.denormalize(
                    outputs[1][rollout_step][self.sample_idx, ..., : pl_module.fcdim], in_place=False
                )
                .squeeze()
                .cpu()
                .numpy(),
            )

            self._output_figure(
                trainer,
                fig,
                tag=f"gnn_pred_ens_val_rstep{rollout_step:02d}_batch{batch_idx:05d}_rank{pl_module.global_rank:02d}",
                exp_log_tag=f"val_pred_ens_rstep{rollout_step:02d}_rank{pl_module.global_rank:02d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # plotting happens only on device 0
        # device 0 has already gathered all ensemble members generated by its group
        if batch_idx % self.plot_frequency == 3 and pl_module.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx)


class InferenceCheckpoint(ModelCheckpoint):
    """A checkpoint callback that saves the model after every validation epoch."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def _torch_drop_down(self, trainer: pl.Trainer) -> torch.nn.Module:
        # Get the model from the DataParallel wrapper
        return trainer.model.module.model if hasattr(trainer.model, "module") else trainer.model.model

    def _sanitise_checkpoints(self, model) -> None:
        # Delete paths from checkpoint
        for path in model.config.hardware.paths.keys():
            model.config.hardware.paths[path] = "/"
        # Delete filenames from checkpoint
        for file in model.config.hardware.files.keys():
            model.config.hardware.files[file] = "/"
        # Disable logging and plotting
        model.config.diagnostics.plot.enabled = False
        model.config.diagnostics.log.wandb.enabled = False
        return model

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        # trainer.save_checkpoint(filepath, self.save_weights_only)

        model = self._torch_drop_down(trainer)
        model = self._sanitise_checkpoints(model)

        torch.save(model, filepath)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            from weakref import proxy

            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


def get_callbacks(config: DictConfig) -> List:
    """Setup callbacks for PyTorch Lightning trainer.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    List
        A list of PyTorch Lightning callbacks
    """
    checkpoint_settings = dict(
        dirpath=config.hardware.paths.checkpoints,
        # monitor="val_wmse",
        verbose=False,
        save_top_k=config.diagnostics.checkpoint.save_top_k,
        # save weights, optimizer states, LR-schedule states, hyperparameters etc.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#contents-of-a-checkpoint
        save_weights_only=False,
        auto_insert_metric_name=False,
        # save after every validation epoch, if we've improved
        save_on_train_epoch_end=False,
        enable_version_counter=True,
    )

    ckpt_frequency_save_dict = dict(
        train_time_interval=timedelta(minutes=config.diagnostics.checkpoint.every_n_minutes)
        if config.diagnostics.checkpoint.every_n_minutes
        else None,
        every_n_epochs=config.diagnostics.checkpoint.every_n_epochs,
        every_n_train_steps=config.diagnostics.checkpoint.every_n_train_steps,
    )

    ckpt_frequency_save_dict = {}
    for key in config.diagnostics.checkpoint.keys():
        if key == "every_n_minutes":
            target = "train_time_interval"
            frequency = timedelta(minutes=config.diagnostics.checkpoint[key])
        else:
            target = key
            frequency = config.diagnostics.checkpoint[key]
        ckpt_frequency_save_dict[target] = (config.hardware.files.checkpoint[key], frequency)

    trainer_callbacks = []
    if not config.diagnostics.profiler:
        for save_key, save_frequency in ckpt_frequency_save_dict.items():
            if save_frequency is not None:
                LOGGER.debug("Checkpoint callback at %s = %s ...", save_key, save_frequency)
                trainer_callbacks.extend(
                    [
                        ModelCheckpoint(
                            filename=config.hardware.files.checkpoint,
                            save_last=True,
                            **{save_key: save_frequency},
                            **checkpoint_settings,
                        ),
                        InferenceCheckpoint(
                            config=config,
                            filename="inference-" + config.hardware.files.checkpoint,
                            save_last=False,
                            **{save_key: save_frequency},
                            **checkpoint_settings,
                        ),
                    ]
                )
            else:
                LOGGER.debug("Not setting up a checkpoint callback with %s", save_key)
    else:
        # the tensorboard logger + pytorch profiler cause pickling errors when writing checkpoints
        LOGGER.warning("Profiling is enabled - AIFS will not write any training or inference model checkpoints!")

    if config.diagnostics.log.wandb.enabled:
        from pytorch_lightning.callbacks import LearningRateMonitor

        trainer_callbacks.append(
            LearningRateMonitor(
                logging_interval="step",
                log_momentum=False,
            )
        )

    if config.diagnostics.eval.enabled:
        trainer_callbacks.append(RolloutEval(config))

    if config.diagnostics.plot.enabled:
        trainer_callbacks.extend(
            [
                RankHistogramPlot(config),
                PredictedEnsemblePlot(config),
                KCRPSMapPlot(config),
                SpreadSkillPlot(config),
            ]
        )
        if config.training.loss == "kcrps":
            trainer_callbacks.append(KCRPSBarPlot(config))
        if config.training.eda_initial_perturbations:
            trainer_callbacks.append(PlotEnsembleInitialConditions(config))

    # if config.diagnostics.plot.learned_features:
    #     LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
    #     trainer_callbacks.append(GraphTrainableFeaturesPlot(config))

    return trainer_callbacks
