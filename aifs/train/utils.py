import argparse
import datetime as dt
import os
from typing import List

import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

from aifs.train.callbacks import RolloutEval
from aifs.utils.config import YAMLConfig
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


def pl_scaling(plev):
    return np.array(plev) / 1000


def setup_wandb_logger(config: YAMLConfig):
    if config["model:wandb:enabled"]:
        from pytorch_lightning.loggers.wandb import WandbLogger

        logger = WandbLogger(
            project="GNN",
            entity="ecmwf-s2s",
            save_dir=os.path.join(
                config["output:basedir"].format(resolution=config["input:resolution"]),
                config["output:logging:log-dir"],
            ),
        )
        logger.log_hyperparams(config.cfg)
        return logger

    LOGGER.warning("You did not set up an experiment logger ...")
    return False


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="YAML configuration file")
    return parser.parse_args()


def setup_callbacks(config: YAMLConfig, timestamp: dt.datetime) -> List:
    trainer_callbacks = [
        # EarlyStopping(monitor="val_wmse", min_delta=0.0, patience=7, verbose=False, mode="min"),
        ModelCheckpoint(
            dirpath=os.path.join(
                config["output:basedir"].format(resolution=config["input:resolution"]),
                config["output:checkpoints:ckpt-dir"],
                timestamp,
            ),
            filename=config["output:model:checkpoint-filename"],
            monitor="val_wmse",
            verbose=False,
            save_top_k=config["output:model:save-top-k"],
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

    if config["diagnostics:eval:enabled"]:
        trainer_callbacks.append(
            RolloutEval(rollout=config["diagnostics:eval:rollout"], frequency=config["diagnostics:eval:frequency"])
        )

    if config["model:swa:enabled"]:
        trainer_callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config["model:swa:lr"],
                swa_epoch_start=min(int(0.75 * config["model:max-epochs"]), config["model:max-epochs"] - 1),
                annealing_epochs=max(int(0.25 * config["model:max-epochs"]), 1),
                annealing_strategy="cos",
                # TODO: do we want the averaging to happen on the CPU, to save memory?
                device=None,
            )
        )

    return trainer_callbacks
