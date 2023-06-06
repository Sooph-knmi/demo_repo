import datetime as dt
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.profilers import AdvancedProfiler

from aifs.data.era_datamodule import ERA5DataModule
from aifs.train.fc_trainer import GraphForecaster
from aifs.train.utils import get_args, setup_wandb_logger, pl_scaling, setup_callbacks
from aifs.utils.config import YAMLConfig
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


def fc_train(config: YAMLConfig) -> None:
    """
    Train entry point.
    Args:
        config: job configuration
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    torch.set_float32_matmul_precision("high")

    # create data module (data loaders and data sets)
    dmod = ERA5DataModule(config)

    # number of variables (features)
    num_features = config["input:num-features"]
    num_aux_features = config["input:num-aux-features"]
    num_fc_features = num_features - num_aux_features

    loss_scaling = np.array([], dtype=np.float32)
    for scl in config["input:loss-scaling-pl"]:
        loss_scaling = np.append(loss_scaling, [scl] * pl_scaling(config["input:pl:levels"]))
    for scl in config["input:loss-scaling-sfc"]:
        loss_scaling = np.append(loss_scaling, [scl])
    assert len(loss_scaling) == num_fc_features
    # LOGGER.debug("Loss scaling: %s", loss_scaling)
    loss_scaling = torch.from_numpy(loss_scaling)

    LOGGER.debug("Total number of prognostic variables: %d", num_fc_features)
    LOGGER.debug("Total number of auxiliary variables: %d", num_aux_features)

    # learning rate multiplier when running single-node, multi-GPU and/or multi-node
    total_gpu_count = config["model:num-nodes"] * config["model:num-gpus"]
    LOGGER.debug("Total GPU count: %d - NB: the learning rate will be scaled by this factor!", total_gpu_count)
    LOGGER.debug("Effective learning rate: %.3e", total_gpu_count * config["model:learn-rate"])

    graph_data = torch.load(
        os.path.join(config["graph:data-basedir"], config["graph:data-file"].format(resolution=config["input:resolution"]))
    )

    fc_model = GraphForecaster(
        graph_data=graph_data,
        metadata=dmod.input_metadata,
        fc_dim=num_fc_features,
        aux_dim=num_aux_features,
        num_levels=len(config["input:pl:levels"]),
        encoder_out_channels=config["model:encoder:num-out-channels"],
        encoder_num_layers=config["model:encoder:num-layers"],
        mlp_extra_layers=config["model:encoder:mlp-extra-layers"],
        activation=config["model:encoder:activation"],
        lr=total_gpu_count * config["model:learn-rate"],
        save_basedir=os.path.join(
            config["output:basedir"].format(resolution=config["input:resolution"]), config["output:plots:plot-dir"], timestamp
        ),
        log_to_wandb=config["model:wandb:enabled"],
        loss_scaling=loss_scaling,
        pl_names=config["input:pl:names"],
        metric_names=config["metrics"],
    )

    if not config["model:warm-restart:enabled"]:
        # load AE checkpoint states
        ae_ckpt = torch.load(config["model:ae:ckpt-path"], map_location=torch.device("cpu"))
        fc_model.load_and_freeze_encoder_weights(ae_ckpt["state_dict"], prefix=config["model:ae:weights-prefix:encoder"])
        # We train the decoder
        # fc_model.load_and_freeze_decoder_weights(ae_ckpt["state_dict"], prefix=config["model:ae:weights-prefix:decoder"])

    # warm restart?
    ckpt_path: Optional[str] = None
    if config["model:warm-restart:enabled"]:
        ckpt_path = os.path.join(
            config["output:basedir"].format(resolution=config["input:resolution"]),
            config["output:checkpoints:ckpt-dir"],
            config["model:warm-restart:ckpt-path"],
        )
        LOGGER.debug("Training will resume from %s ...", ckpt_path)

    trainer_callbacks = setup_callbacks(config, timestamp)

    if config["model:profile"]:
        profiler = AdvancedProfiler(
            dirpath=os.path.join(
                config["output:basedir"].format(resolution=config["input:resolution"]), config["output:logging:log-dir"]
            ),
            filename="aifs-profiler",
        )
    else:
        profiler = None

    logger = setup_wandb_logger(config)
    if logger and (config["model:log-gradients"] or config["model:log-parameters"]):
        if config["model:log-gradients"] and config["model:log-parameters"]:
            log_ = "all"
        elif config["model:log-gradients"]:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(fc_model, log=log_, log_freq=config["output:logging:log-interval"], log_graph=False)

    trainer = pl.Trainer(
        accelerator="gpu" if config["model:num-gpus"] > 0 else "cpu",
        callbacks=trainer_callbacks,
        detect_anomaly=config["model:debug:anomaly-detection"],
        strategy=config["model:strategy"],  # we should use ddp with find_unused_parameters = False, static_graph = True
        devices=config["model:num-gpus"] if config["model:num-gpus"] > 0 else None,
        num_nodes=config["model:num-nodes"],
        precision=config["model:precision"],
        max_epochs=config["model:max-epochs"],
        logger=setup_wandb_logger(config),
        log_every_n_steps=config["output:logging:log-interval"],
        # run a fixed no of batches per epoch (helpful when debugging)
        limit_train_batches=config["model:limit-batches:training"],
        limit_val_batches=config["model:limit-batches:validation"],
        num_sanity_val_steps=0,
        # we have our own DDP-compliant sampler logic baked into the dataset
        use_distributed_sampler=False,
        profiler=profiler,
    )

    trainer.fit(fc_model, datamodule=dmod, ckpt_path=ckpt_path)
    LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)
    fc_train(config)