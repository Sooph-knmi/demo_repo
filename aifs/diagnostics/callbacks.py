import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from aifs.diagnostics.plots import init_plot_settings
from aifs.diagnostics.plots import plot_graph_features
from aifs.diagnostics.plots import plot_loss
from aifs.diagnostics.plots import plot_predicted_multilevel_flat_sample
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

    def _output_figure(self, trainer, fig, epoch: int, tag: str = "gnn", exp_log_tag: str = "val_pred_sample") -> None:
        """Figure output: save to file and/or display in notebook."""
        if self.save_basedir is not None:
            save_path = Path(
                self.save_basedir,
                "plots",
                f"{tag}_epoch{epoch:03d}.png",
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            if self.config.diagnostics.log.wandb.enabled:
                import wandb

                trainer.logger.experiment.log({exp_log_tag: wandb.Image(fig)})
        plt.close(fig)  # cleanup


class AsyncPlotCallback(PlotCallback):
    """Factory for creating a callback that plots data to Weights and Biases."""

    def __init__(self, config):
        super().__init__(config)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._error: Optional[BaseException] = None

    def teardown(self, trainer, pl_module, stage) -> None:
        """This method is called to close the threads."""
        self._executor.shutdown(wait=True)

        # if an error was raised anytime in any of the `executor.submit` calls
        if self._error:
            raise self._error


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
        self.rollout = config.diagnostics.eval.rollout
        self.frequency = config.diagnostics.eval.frequency

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        # NB! the batch is already normalized in-place - see pl_model.validation_step()
        metrics = {}

        # start rollout
        x = batch[:, 0 : pl_module.multi_step, ..., pl_module.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)
        assert batch.shape[1] >= self.rollout + pl_module.multi_step, "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            for rstep in range(self.rollout):
                y_pred = pl_module(x)  # prediction at rollout step rstep, shape = (bs, latlon, nvar)
                y = batch[
                    :, pl_module.multi_step + rstep, ..., pl_module.data_indices.data.output.full
                ]  # target, shape = (bs, latlon, nvar)
                # y includes the auxiliary variables, so we must leave those out when computing the loss
                loss += pl_module.loss(y_pred, y)

                x = pl_module.advance_input(batch, y_pred)

                y_denorm = pl_module.model.normalizer.denormalize(
                    y, in_place=False, data_index=pl_module.data_indices.data.output.full
                )
                y_pred_denorm = pl_module.model.normalizer.denormalize(
                    x[:, -1, ...], in_place=False, data_index=pl_module.data_indices.data.output.full
                )
                for mkey, indices in pl_module.metric_ranges.items():
                    metrics[f"{mkey}_{rstep+1}"] = pl_module.metrics(y_pred_denorm[..., indices], y_denorm[..., indices])

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
            logger=pl_module.logger_enabled,
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
                logger=pl_module.logger_enabled,
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

    def _plot(
        # self, trainer, latlons:np.ndarray, features:np.ndarray, tag:str, exp_log_tag:str
        self,
        trainer,
        latlons,
        features,
        epoch,
        tag,
        exp_log_tag,
    ) -> None:
        fig = plot_graph_features(latlons, features)
        self._output_figure(trainer, fig, epoch=epoch, tag=tag, exp_log_tag=exp_log_tag)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.global_rank == 0:
            model = pl_module.model.module.model if hasattr(pl_module.model, "module") else pl_module.model.model
            graph = pl_module.graph_data.cpu()
            epoch = trainer.current_epoch

            if model.era_trainable is not None:
                ecoords = np.rad2deg(graph[("era", "to", "era")].ecoords_rad.numpy())

                self._plot(
                    trainer,
                    ecoords,
                    model.era_trainable.cpu(),
                    epoch=epoch,
                    tag="era_trainable",
                    exp_log_tag="era_trainable",
                )

            if model.h_trainable is not None:
                hcoords = np.rad2deg(graph[("h", "to", "h")].hcoords_rad.numpy())
                self._plot(trainer, hcoords, model.h_trainable.cpu(), epoch=epoch, tag="h_trainable", exp_log_tag="h_trainable")


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
        epoch,
    ) -> None:
        for rollout_step in range(pl_module.rollout):
            y_hat = outputs[1][rollout_step]
            y_true = batch[:, pl_module.multi_step + rollout_step, :, pl_module.data_indices.data.output.full]
            loss = pl_module.loss(y_hat, y_true, squash=False).cpu().numpy()

            fig = plot_loss(loss)
            self._output_figure(
                trainer,
                fig,
                epoch=epoch,
                tag=f"loss_rstep_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
                exp_log_tag=f"loss_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, epoch=trainer.current_epoch)


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
        epoch,
    ) -> None:
        # Build dictionary of inidicies and parameters to be plotted
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: (name, name not in self.config.data.diagnostic)
            for name in self.config.diagnostics.plot.parameters
        }

        data = (
            pl_module.model.normalizer.denormalize(
                batch[
                    self.sample_idx,
                    pl_module.multi_step - 1 : pl_module.multi_step + pl_module.rollout + 1,
                    ...,
                    pl_module.data_indices.data.output.full,
                ],
                in_place=False,
                data_index=pl_module.data_indices.data.output.full,
            )
            .cpu()
            .numpy()
        )

        latlons = np.rad2deg(pl_module.data_latlons.cpu().numpy())
        for rollout_step in range(pl_module.rollout):
            fig = plot_predicted_multilevel_flat_sample(
                plot_parameters_dict,
                self.config.diagnostics.plot.per_sample,
                latlons,
                data[0, ...].squeeze(),
                data[rollout_step + 1, ...].squeeze(),
                pl_module.model.normalizer.denormalize(
                    outputs[1][rollout_step][self.sample_idx, ...],
                    in_place=False,
                    data_index=pl_module.data_indices.data.output.full,
                )
                .squeeze()
                .cpu()
                .numpy(),
            )

            self._output_figure(
                trainer,
                fig,
                epoch=epoch,
                tag=f"gnn_pred_val_sample_rstep{rollout_step:02d}_batch{batch_idx:04d}_rank0",
                exp_log_tag=f"val_pred_sample_rstep{rollout_step:02d}_rank{pl_module.local_rank:01d}",
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.plot_frequency == 3 and trainer.global_rank == 0:
            self._plot(trainer, pl_module, outputs, batch, batch_idx, epoch=trainer.current_epoch)


class InferenceCheckpoint(ModelCheckpoint):
    """A checkpoint callback that saves the model after every validation epoch."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def _torch_drop_down(self, trainer: pl.Trainer) -> torch.nn.Module:
        # Get the model from the DataParallel wrapper, for single and multi-gpu cases
        assert hasattr(trainer, "model"), "Trainer has no attribute 'model'! Is the Pytorch Lightning version correct?"
        return trainer.model.module.model if hasattr(trainer.model, "module") else trainer.model.model

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        if not trainer.is_global_zero:
            return

        # trainer.save_checkpoint(filepath, self.save_weights_only)

        model = self._torch_drop_down(trainer)

        save_config = model.config
        model.config = None

        save_metadata = model.metadata
        model.metadata = None

        torch.save(model, filepath)

        with ZipFile(filepath, "a") as zipf:
            base, _ = os.path.splitext(os.path.basename(filepath))
            zipf.writestr(
                f"{base}/ai-models.json",
                json.dumps(save_metadata, indent=4),
            )

        model.config = save_config
        model.metadata = save_metadata

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
        verbose=False,
        # save weights, optimizer states, LR-schedule states, hyperparameters etc.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#contents-of-a-checkpoint
        save_weights_only=False,
        auto_insert_metric_name=False,
        # save after every validation epoch, if we've improved
        save_on_train_epoch_end=False,
        enable_version_counter=False,
        save_top_k=-1,
    )

    ckpt_frequency_save_dict = {}
    for key, frequency in config.diagnostics.checkpoint.items():
        if key == "every_n_minutes":
            target = "train_time_interval"
            frequency = timedelta(minutes=frequency)
        else:
            target = key
        ckpt_frequency_save_dict[target] = (config.hardware.files.checkpoint[key], frequency)

    trainer_callbacks = []
    if not config.diagnostics.profiler:
        for save_key, (name, save_frequency) in ckpt_frequency_save_dict.items():
            if save_frequency is not None:
                LOGGER.debug("Checkpoint callback at %s = %s ...", save_key, save_frequency)
                trainer_callbacks.extend(
                    [
                        ModelCheckpoint(
                            filename=name,
                            save_last=True,
                            **{save_key: save_frequency},
                            **checkpoint_settings,
                        ),
                        InferenceCheckpoint(
                            config=config,
                            filename="inference-" + name,
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
                PlotLoss(config),
                PlotSample(config),
            ]
        )

    if config.training.swa.enabled:
        from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

        trainer_callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.training.swa.lr,
                swa_epoch_start=min(
                    int(0.75 * config.training.max_epochs),
                    config.training.max_epochs - 1,
                ),
                annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
                annealing_strategy="cos",
                # TODO: do we want the averaging to happen on the CPU, to save memory?
                device=None,
            )
        )

    if config.diagnostics.plot.learned_features:
        LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
        trainer_callbacks.append(GraphTrainableFeaturesPlot(config))

    return trainer_callbacks
