import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def get_wandb_logger(config: DictConfig, model: pl.LightningModule):
    """Setup Weights & Biases experiment logger.

    Parameters
    ----------
    config : DictConfig
        Job configuration
    model: GraphForecaster
        Model to watch

    Returns
    -------
    _type_
        Logger object or False
    """
    if not config.diagnostics.logging.wandb.enabled:
        LOGGER.debug("Weights & Biases logging is disabled.")
        return None

    from pytorch_lightning.loggers.wandb import WandbLogger

    logger = WandbLogger(
        project="aifs-fc",
        entity="ecmwf-ml",
        id=config.training.run_id,
        save_dir=config.hardware.paths.logs,
        offline=config.diagnostics.logging.wandb.offline,
        log_model=config.diagnostics.logging.wandb.log_model,
        resume=config.training.run_id is not None,
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    if config.diagnostics.logging.wandb.gradients or config.diagnostics.logging.wandb.parameters:
        if config.diagnostics.logging.wandb.gradients and config.diagnostics.logging.wandb.parameters:
            log_ = "all"
        elif config.diagnostics.logging.wandb.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=config.diagnostics.logging.interval, log_graph=False)

    return logger
