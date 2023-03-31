import os
import argparse

# import mlflow
from gnn_era5.utils.config import YAMLConfig
from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__)


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

    # if config["model:mlflow:enabled"]:
    #     from pytorch_lightning.loggers.mlflow import MLFlowLogger

    #     run = mlflow.active_run()
    #     return MLFlowLogger(
    #         experiment_name="gnn-era5",
    #         tracking_uri="https://mlflow.copernicus-climate.eu",
    #         run_id=run.info.run_id,
    #     )
    # if config["model:aim:enabled"]:
    #     # make sure you `pip install aim` first
    #     from aim.pytorch_lightning import AimLogger

    #     # track experimental data by using Aim
    #     return AimLogger(
    #         repo="aim://marsdev-core:5555",
    #         experiment="gnn-era5",
    #         train_metric_prefix="train_",
    #         val_metric_prefix="val_",
    #     )
    # if config["model:comet:enabled"]:
    #     from pytorch_lightning.loggers import CometLogger

    #     return CometLogger(
    #         project_name="gnn-era5",
    #         workspace="mchantry",
    #     )


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="YAML configuration file")
    return parser.parse_args()
