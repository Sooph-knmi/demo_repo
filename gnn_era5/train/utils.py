import os
import argparse
import numpy as np

# import mlflow
from gnn_era5.utils.config import YAMLConfig
from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__)


def pl_scaling(plev):
    return np.array(plev) / 1000


def setup_exp_logger(config: YAMLConfig):
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
        logger.log_hyperparams(config._cfg)
        return logger
    if config["model:neptune:enabled"]:
        from pytorch_lightning.loggers.neptune import NeptuneLogger

        logger = NeptuneLogger(
            project="ecmwf/gnn-era5",
            log_model_checkpoints=False,
        )
        logger.log_hyperparams(config._cfg)
        return logger

    LOGGER.warning("You did not set up an experiment logger ...")
    return False


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="YAML configuration file")
    return parser.parse_args()
