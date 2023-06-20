import argparse
import datetime as dt
import os
from typing import List

import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

from aifs.train.callbacks import RolloutEval
# from aifs.utils.config import YAMLConfig
from aifs.utils.logger import get_logger

from omegaconf import OmegaConf, DictConfig

LOGGER = get_logger(__name__)


def pl_scaling(plev):
    return np.array(plev) / 1000


def setup_wandb_logger(config: DictConfig):
    if config.diagnostics.logging.wandb:
        from pytorch_lightning.loggers.wandb import WandbLogger

        logger = WandbLogger(
            project="aifs-fc",
            entity="ecmwf-ml",
            save_dir=config.paths.logs,
        )
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
        return logger

    LOGGER.warning("You did not set up an experiment logger ...")
    return False

def setup_callbacks(config: DictConfig, timestamp: dt.datetime) -> List:
    trainer_callbacks = [
        # EarlyStopping(monitor="val_wmse", min_delta=0.0, patience=7, verbose=False, mode="min"),
        ModelCheckpoint(
            dirpath=os.path.join(
                config.paths.checkpoints,
                timestamp,
            ),
            filename=config.files.checkpoint,
            monitor="val_wmse",
            verbose=False,
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
        trainer_callbacks.append(
            RolloutEval(rollout=config.diagnostics.eval.rollout, frequency=config.diagnostics.eval.frequency)
        )

    if config.training.swa.enabled:
        trainer_callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.training.swa.lr,
                swa_epoch_start=min(int(0.75 * config.training.max_epochs), config.training.max_epochs - 1),
                annealing_epochs=max(int(0.25 * config.training.max_epochs), 1),
                annealing_strategy="cos",
                # TODO: do we want the averaging to happen on the CPU, to save memory?
                device=None,
            )
        )

    return trainer_callbacks
