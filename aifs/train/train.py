import os
import sys
from typing import Optional

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


def train(config: DictConfig) -> None:
    from aifs.data.era_datamodule import ERA5DataModule
    from aifs.train.trainer import GraphForecaster
    from aifs.train.utils import setup_callbacks
    from aifs.train.utils import setup_wandb_logger
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.profilers import AdvancedProfiler

    """Training entry point.

    Parameters
    ----------
    config : DictConfig
        Job configuration
    """
    torch.set_float32_matmul_precision("high")

    # create data module (data loaders and data sets)
    dmod = ERA5DataModule(config)

    # number of variables (features)
    num_features = config.data.num_features
    num_aux_features = config.data.num_aux_features
    num_fc_features = num_features - num_aux_features

    LOGGER.debug("Total number of prognostic variables: %d", num_fc_features)
    LOGGER.debug("Total number of auxiliary variables: %d", num_aux_features)

    # learning rate multiplier when running single-node, multi-GPU and/or multi-node
    total_gpu_count = config.hardware.num_nodes * config.hardware.num_gpus_per_node
    LOGGER.debug("Total GPU count: %d - NB: the learning rate will be scaled by this factor!", total_gpu_count)
    LOGGER.debug("Effective learning rate: %.3e", total_gpu_count * config.training.lr.rate)
    LOGGER.debug("Rollout window length: %d", config.training.rollout.start)

    model = GraphForecaster(metadata=dmod.input_metadata, config=config)

    if config.training.compile:
        # this doesn't work ATM (April 2), don't bother enabling it ...
        LOGGER.debug("torch.compiling the Lightning model ...")
        model = torch.compile(model, mode="default", backend="inductor", fullgraph=False)

    # warm restart?
    ckpt_path: Optional[str] = None
    if config.hardware.files.warm_start:
        ckpt_path = os.path.join(
            config.hardware.paths.checkpoints,
            config.hardware.files.warm_start,
        )
        LOGGER.debug("Training will resume from %s ...", ckpt_path)

    trainer_callbacks = setup_callbacks(config, config.hardware.paths.run_id)

    if config.diagnostics.profiler:
        profiler = AdvancedProfiler(
            dirpath=config.hardware.paths.logs,
            filename="aifs-profiler",
        )
    else:
        profiler = None

    logger = setup_wandb_logger(config)
    if logger and (config.diagnostics.logging.gradients or config.diagnostics.logging.parameters):
        if config.diagnostics.logging.gradients and config.diagnostics.logging.parameters:
            log_ = "all"
        elif config.diagnostics.logging.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=config.diagnostics.logging.interval, log_graph=False)

    trainer = pl.Trainer(
        accelerator="gpu" if config.hardware.num_gpus_per_node > 0 else "cpu",
        callbacks=trainer_callbacks,
        detect_anomaly=config.diagnostics.debug.anomaly_detection,
        strategy=config.hardware.strategy,  # we should use ddp with find_unused_parameters = False, static_graph = True
        devices=config.hardware.num_gpus_per_node if config.hardware.num_gpus_per_node > 0 else None,
        num_nodes=config.hardware.num_nodes,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        logger=setup_wandb_logger(config),
        log_every_n_steps=config.diagnostics.logging.interval,
        # run a fixed no of batches per epoch (helpful when debugging)
        limit_train_batches=config.dataloader.limit_batches.training,
        limit_val_batches=config.dataloader.limit_batches.validation,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.training.accum_grad_batches,
        gradient_clip_val=config.training.gradient_clip.val,
        gradient_clip_algorithm=config.training.gradient_clip.algorithm,
        # we have our own DDP-compliant sampler logic baked into the dataset
        use_distributed_sampler=False,
        profiler=profiler,
    )

    trainer.fit(model, datamodule=dmod, ckpt_path=ckpt_path)
    LOGGER.debug("---- DONE. ----")


def hydra_main_with_overwrite(*args, **kwargs):
    def pop_argument(key):
        """Finds argument --{key} in sys.argv and remove it.

        Returns the value found, or None if --{key} is not present
        """
        try:
            i = sys.argv.index(f"--{key}")
        except ValueError:
            return None
        assert len(sys.argv) > i, f"--{key} requires an argument."
        assert sys.argv.pop(i) == f"--{key}"
        return sys.argv.pop(i)  # get value

    def decorator(func):
        overwrite = pop_argument("overwrite")

        @hydra.main(*args, **kwargs)
        def wrapper(config: DictConfig):
            if overwrite:
                LOGGER.info(f"Using {overwrite} to overwrite config.")
                overwrite_cfg = OmegaConf.load(overwrite)
                config = OmegaConf.merge(config, overwrite_cfg)
            return func(config)

        return wrapper

    return decorator


@hydra_main_with_overwrite(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    train(config)
