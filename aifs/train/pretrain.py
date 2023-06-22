import os
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.profilers import AdvancedProfiler

from aifs.data.era_datamodule import ERA5DataModule
from aifs.diagnostics.logger import get_logger
from aifs.train.pretrainer import CriticPretrainer
from aifs.train.pretrainer import GeneratorPretrainer
from aifs.train.utils import setup_callbacks
from aifs.train.utils import setup_wandb_logger

LOGGER = get_logger(__name__)


def pretrain(config: DictConfig) -> None:
    """Entry point for pre-training the generator or discriminator.

    Parameters
    ----------
    config : DictConfig
        Job configuration
    """
    torch.set_float32_matmul_precision("high")

    if config.training.initial_seed is not None:
        initial_seed = config.training.initial_seed
        pl.seed_everything(initial_seed, workers=True)
        LOGGER.debug("Running with initial seed: %d", initial_seed)

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

    ckpt_path: Optional[str] = None
    if config.training.pretrain.generator:
        LOGGER.info("Pretraining the generator ...")
        LOGGER.debug("Effective generator learning rate: %.3e", total_gpu_count * config.training.lr.rate.generator)
        if config.hardware.files.warm_start.generator is not None:
            ckpt_path = os.path.join(
                config.hardware.paths.checkpoints,
                config.hardware.files.warm_start.generator,
            )
            LOGGER.debug("Generator pre-training will resume from %s ...", ckpt_path)
        model = GeneratorPretrainer(metadata=dmod.input_metadata, config=config)
        trainer_callbacks = setup_callbacks(config, config.hardware.paths.run_id, config.hardware.files.checkpoint.generator)
    elif config.training.pretrain.critic:
        LOGGER.info("Pretraining the critic ...")
        LOGGER.debug("Effective critic learning rate: %.3e", total_gpu_count * config.training.lr.rate.critic)
        if config.hardware.files.warm_start.critic is not None:
            ckpt_path = os.path.join(
                config.hardware.paths.checkpoints,
                config.hardware.files.warm_start.critic,
            )
            LOGGER.debug("Critic pre-training will resume from %s ...", ckpt_path)
        if config.hardware.files.warm_start.generator is None:
            LOGGER.error("The generator weights MUST be initialized from a checkpoint - check your hardware config ...")
            raise RuntimeError
        model = CriticPretrainer(metadata=dmod.input_metadata, config=config)
        trainer_callbacks = setup_callbacks(config, config.hardware.paths.run_id, config.hardware.files.checkpoint.critic)
    else:
        raise RuntimeError("Pretraining of either the generator or critic should be enabled in your training config file!")

    LOGGER.debug("Rollout window length: %d", config.training.rollout.start)

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
        deterministic=config.training.deterministic,
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
        # we have our own DDP-compliant sampler logic baked into the dataset
        use_distributed_sampler=False,
        profiler=profiler,
    )

    trainer.fit(model, datamodule=dmod, ckpt_path=ckpt_path)
    LOGGER.debug("---- DONE. ----")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    pretrain(config)
