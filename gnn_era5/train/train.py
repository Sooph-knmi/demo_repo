import datetime as dt
import os

import torch
import pytorch_lightning as pl

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from gnn_era5.utils.config import YAMLConfig
from gnn_era5.data.era_datamodule import ERA5DataModule
from gnn_era5.utils.logger import get_logger
from gnn_era5.train.trainer import GraphForecaster
from gnn_era5.train.utils import setup_exp_logger, get_args

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
    num_features = len(config["input:pl:names"]) * len(config["input:pl:levels"]) + len(config["input:sfc:names"])
    num_aux_features = config["input:num-aux-features"]
    num_fc_features = num_features - num_aux_features

    loss_scaling = []
    for scl in config["input:loss-scaling-pl"]:
        loss_scaling.extend([scl] * len(config["input:pl:levels"]))
    for scl in config["input:loss-scaling-sfc"]:
        loss_scaling.append(scl)
    assert len(loss_scaling) == num_fc_features
    print(loss_scaling)
    loss_scaling = torch.Tensor(loss_scaling)

    LOGGER.debug("Total number of prognostic variables: %d", num_fc_features)
    LOGGER.debug("Total number of auxiliary variables: %d", num_aux_features)

    # learning rate multiplier when running single-node, multi-GPU and/or multi-node
    total_gpu_count = config["model:num-nodes"] * config["model:num-gpus"]
    LOGGER.debug("Total GPU count: %d - NB: the learning rate will be scaled by this factor!", total_gpu_count)
    LOGGER.debug("Effective learning rate: %.3e", total_gpu_count * config["model:learn-rate"])
    LOGGER.debug("Rollout window length: %d", config["model:rollout"])

    graph_data = torch.load(os.path.join(config["graph:data-basedir"], config["graph:data-file"]))

    # TODO: revisit this?
    # all weights = 1
    # var_dt_sigma = np.ones(len(config["input:variables:names"]) * len(config["input:variables:levels"]))

    model = GraphForecaster(
        graph_data=graph_data,
        fc_dim=num_fc_features,
        aux_dim=num_aux_features,
        encoder_hidden_channels=config["model:encoder:num-hidden-channels"],
        encoder_out_channels=config["model:encoder:num-out-channels"],
        # encoder_dropout=config["model:encoder:dropout"],
        encoder_num_layers=config["model:encoder:num-layers"],
        encoder_mapper_num_layers=config["model:encoder:mapper-num-layers"],
        # encoder_num_heads=config["model:encoder:num-heads"],
        # encoder_activation=config["model:encoder:activation"],
        # use_dynamic_context=True,
        lr=total_gpu_count * config["model:learn-rate"],
        rollout=config["model:rollout"],
        save_basedir=os.path.join(
            config["output:basedir"].format(resolution=config["input:resolution"]), config["output:plots:plot-dir"], timestamp
        ),
        log_to_wandb=config["model:wandb:enabled"],
        log_to_neptune=config["model:neptune:enabled"],
        loss_scaling=loss_scaling,
    )

    if config["model:compile"]:
        # TODO: is it better if we compile smaller chunks of the model - like the msg passing MLPs?
        LOGGER.debug("torch.compiling the Lightning model ...")
        model = torch.compile(model, mode="default", backend="inductor", fullgraph=False)

    trainer = pl.Trainer(
        accelerator="gpu" if config["model:num-gpus"] > 0 else "cpu",
        callbacks=[
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
                save_weights_only=True,
                mode="min",
                auto_insert_metric_name=True,
                save_on_train_epoch_end=True,
                every_n_epochs=1,
            )
        ],
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
        # we have our own DDP-compliant sampler logic baked into the dataset
        # I'm running with lightning 2.0, if you use an older version comment out the following line
        # and use `replace_sampler_ddp=False` instead
        use_distributed_sampler=False,
        # replace_sampler_ddp=False,
    )

    trainer.fit(model, datamodule=dmod)

    LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)
    train(config)
