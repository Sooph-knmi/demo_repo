import datetime as dt
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

from gnn_era5.data.era_datamodule import ERA5DataModule
from gnn_era5.train.trainer import GraphForecaster
from gnn_era5.train.utils import get_args, setup_exp_logger, pl_scaling
from gnn_era5.utils.config import YAMLConfig
from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__)


def train(config: YAMLConfig) -> None:
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
    LOGGER.debug("Rollout window length: %d", config["model:rollout"])

    graph_data = torch.load(
        os.path.join(config["graph:data-basedir"], config["graph:data-file"].format(resolution=config["input:resolution"]))
    )

    model = GraphForecaster(
        graph_data=graph_data,
        metadata=dmod.input_metadata,
        fc_dim=num_fc_features,
        aux_dim=num_aux_features,
        num_levels=len(config["input:pl:levels"]),
        encoder_hidden_channels=config["model:encoder:num-hidden-channels"],
        encoder_out_channels=config["model:encoder:num-out-channels"],
        encoder_num_layers=config["model:encoder:num-layers"],
        encoder_mapper_num_layers=config["model:encoder:mapper-num-layers"],
        activation=config["model:encoder:activation"],
        lr=total_gpu_count * config["model:learn-rate"],
        rollout=config["model:rollout"],
        save_basedir=os.path.join(
            config["output:basedir"].format(resolution=config["input:resolution"]), config["output:plots:plot-dir"], timestamp
        ),
        act_checkpoints=config["model:act-checkpoints"],
        log_to_wandb=config["model:wandb:enabled"],
        log_to_neptune=config["model:neptune:enabled"],
        log_persistence=False,
        loss_scaling=loss_scaling,
        pl_names=config["input:pl:names"],
        metric_names=config["metrics"],
    )

    if config["model:compile"]:
        # this doesn't work ATM (April 2), don't bother enabling it ...
        LOGGER.debug("torch.compiling the Lightning model ...")
        model = torch.compile(model, mode="default", backend="inductor", fullgraph=False)

    # warm restart?
    ckpt_path: Optional[str] = None
    if config["model:warm-restart:enabled"]:
        ckpt_path = os.path.join(
            config["output:basedir"].format(resolution=config["input:resolution"]),
            config["output:checkpoints:ckpt-dir"],
            config["model:warm-restart:ckpt-path"],
        )
        LOGGER.debug("Training will resume from %s ...", ckpt_path)

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
    ]

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

    trainer = pl.Trainer(
        accelerator="gpu" if config["model:num-gpus"] > 0 else "cpu",
        callbacks=trainer_callbacks,
        detect_anomaly=config["model:debug:anomaly-detection"],
        strategy=config["model:strategy"],
        devices=config["model:num-gpus"] if config["model:num-gpus"] > 0 else None,
        num_nodes=config["model:num-nodes"],
        precision=config["model:precision"],
        max_epochs=config["model:max-epochs"],
        logger=setup_exp_logger(config),
        log_every_n_steps=config["output:logging:log-interval"],
        # run a fixed no of batches per epoch (helpful when debugging)
        limit_train_batches=config["model:limit-batches:training"],
        limit_val_batches=config["model:limit-batches:validation"],
        num_sanity_val_steps=0,
        accumulate_grad_batches=config["model:accum-grad-batches"],
        # we have our own DDP-compliant sampler logic baked into the dataset
        # I'm running with lightning 2.0, if you use an older version comment out the following line
        # and use `replace_sampler_ddp=False` instead
        use_distributed_sampler=False,
        # replace_sampler_ddp=False,
    )

    trainer.fit(model, datamodule=dmod, ckpt_path=ckpt_path)

    LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)
    train(config)
