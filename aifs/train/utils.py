import os
from typing import List

import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from aifs.diagnostics.logger import get_logger
from aifs.train.callbacks import GraphTrainableFeaturesPlot
from aifs.train.callbacks import KCRPSPlot
from aifs.train.callbacks import PredictedEnsemblePlot
from aifs.train.callbacks import RankHistogramPlot
from aifs.train.callbacks import RolloutEval
from aifs.train.callbacks import SpreadSkillPlot

# from aifs.train.callbacks import PlotLoss
# from aifs.train.callbacks import PlotSample

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
            project="aifs-ens",
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
            monitor="val_kcrps_epoch",
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
                RankHistogramPlot(config),
                GraphTrainableFeaturesPlot(config),
                PredictedEnsemblePlot(config),
                KCRPSPlot(config),
                SpreadSkillPlot(config),
            ]
        )

    return trainer_callbacks
