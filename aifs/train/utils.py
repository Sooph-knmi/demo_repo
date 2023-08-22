import os
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

from aifs.diagnostics.logger import get_logger
from aifs.train.callbacks import GraphTrainableFeaturesPlot
from aifs.train.callbacks import PlotLoss
from aifs.train.callbacks import PlotSample
from aifs.train.callbacks import RolloutEval

LOGGER = get_logger(__name__)


def pl_scaling(plev):
    """Convert pressure levels to PyTorch Lightning scaling."""
    return np.array(plev) / 1000


def setup_wandb_logger(config: DictConfig):
    """Setup Weights & Biases experiment logger.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    _type_
        Logger object or False
    """
    if config.diagnostics.logging.wandb:
        from pytorch_lightning.loggers.wandb import WandbLogger

        logger = WandbLogger(
            project="aifs-fc",
            entity="ecmwf-ml",
            save_dir=config.hardware.paths.logs,
        )
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
        return logger

    LOGGER.warning("You did not set up an experiment logger ...")
    return False


def setup_callbacks(config: DictConfig, timestamp: str) -> List:
    """Setup callbacks for PyTorch Lightning trainer.

    Parameters
    ----------
    config : DictConfig
        Job configuration
    timestamp : str
        Timestamp of the job

    Returns
    -------
    List
        _description_
    """

    trainer_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(
                config.hardware.paths.checkpoints,
                timestamp,
            ),
            filename=config.hardware.files.checkpoint,
            monitor="val_wmse",
            verbose=False,
            save_last=True,
            save_top_k=config.training.save_top_k,
            # save weights, optimizer states, LR-schedule states, hyperparameters etc.
            # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#contents-of-a-checkpoint
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=True,
            # save after every validation epoch, if we've improved
            save_on_train_epoch_end=False,
            every_n_epochs=1,
        ),
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=True,
        ),
    ]

    if config.diagnostics.eval.enabled:
        trainer_callbacks.append(RolloutEval(config))

    if config.diagnostics.plot.enabled:
        trainer_callbacks.extend(
            [
                PlotLoss(config),
                PlotSample(config),
            ]
        )

    if config.diagnostics.plot.enabled:
        trainer_callbacks.extend(
            [
                PlotLoss(config),
                PlotSample(config),
            ]
        )

    if config.training.swa.enabled:
        assert not config.training.swag.enabled, "Can't enable both SWA and SWAG at the same time! Check your config."
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
                device=torch.device("cpu"),
            )
        )

    if config.diagnostics.plot.learned_features:
        LOGGER.debug("Setting up a callback to plot the trainable graph node features ...")
        trainer_callbacks.append(GraphTrainableFeaturesPlot(config))

    return trainer_callbacks
