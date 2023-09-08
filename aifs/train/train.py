from functools import cached_property
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.profilers import AdvancedProfiler

from aifs.data.era_datamodule import ERA5DataModule
from aifs.diagnostics.callbacks import get_callbacks
from aifs.diagnostics.logging import get_wandb_logger
from aifs.train.forecaster import GraphForecaster
from aifs.train.strategy import DDPGroupStrategy
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class AIFSTrainer:
    """Utility class for training the model."""

    def __init__(self, config: DictConfig):
        # Sets the internal precision of float32 matrix multiplications.
        torch.set_float32_matmul_precision("high")
        # Resolve the config to avoid shenanigans with lazy loading
        OmegaConf.resolve(config)
        self.config = config

        # Default to not warm-starting from a checkpoint
        self.start_from_checkpoint = bool(self.config.training.run_id) or bool(self.config.training.fork_run_id)
        self.config.training.run_id = self.run_id

        # Update paths to contain the run ID
        self.update_paths()

        self.log_information()

    @cached_property
    def datamodule(self) -> ERA5DataModule:
        """DataModule instance and DataSets."""
        return ERA5DataModule(self.config)

    @cached_property
    def initial_seed(self) -> int:
        """Initial seed for the RNG."""
        initial_seed = pl.seed_everything(self.config.training.initial_seed, workers=True)
        LOGGER.debug("Running with initial seed: %d", initial_seed)
        return initial_seed

    @cached_property
    def model(self) -> GraphForecaster:
        """Provide the model instance."""
        return GraphForecaster(metadata=self.datamodule.input_metadata, config=self.config)

    @cached_property
    def run_id(self) -> str:
        """Unique identifier for the current run."""

        if self.config.training.run_id and not self.config.training.fork_run_id:
            # Return the provided run ID
            return self.config.training.run_id

        # Generate a random UUID
        import uuid

        return str(uuid.uuid4())

    @cached_property
    def wandb_logger(self) -> pl.loggers.WandbLogger:
        """WandB logger."""
        return get_wandb_logger(self.config, self.model)

    @cached_property
    def last_checkpoint(self) -> Optional[str]:
        """Path to the last checkpoint."""
        if not self.start_from_checkpoint:
            return None

        checkpoint = Path(
            self.config.hardware.paths.checkpoints.parent,
            self.config.training.fork_run_id or self.run_id,
            self.config.hardware.files.warm_start or "last.ckpt",
        )

        # Check if the last checkpoint exists
        if Path(checkpoint).exists():
            LOGGER.info("Resuming training from last checkpoint: %s", checkpoint)
            return checkpoint
        LOGGER.warning("Could not find last checkpoint: %s", checkpoint)

    @cached_property
    def callbacks(self) -> List[pl.callbacks.Callback]:
        return get_callbacks(self.config)

    @cached_property
    def profiler(self) -> Optional[AdvancedProfiler]:
        if self.config.diagnostics.profiler:
            return AdvancedProfiler(
                dirpath=self.config.hardware.paths.logs,
                filename="aifs-profiler",
            )

    @cached_property
    def loggers(self) -> List:
        loggers = []
        if self.config.diagnostics.logging.wandb.enabled:
            loggers.append(self.wandb_logger)
        return loggers

    @cached_property
    def accelerator(self) -> str:
        return "gpu" if self.config.hardware.num_gpus_per_node > 0 else "cpu"

    def log_information(self) -> None:
        # Log number of variables (features)
        num_fc_features = self.config.data.num_features - self.config.data.num_aux_features
        LOGGER.debug("Total number of prognostic variables: %d", num_fc_features)
        LOGGER.debug("Total number of auxiliary variables: %d", self.config.data.num_aux_features)

        # Log learning rate multiplier when running single-node, multi-GPU and/or multi-node
        total_gpu_count = self.config.hardware.num_nodes * self.config.hardware.num_gpus_per_node
        LOGGER.debug("Number of GPUs per group: %d", self.config.hardware.group_size)
        LOGGER.debug("Total GPU count: %d - NB: the learning rate will be scaled by this factor!", total_gpu_count)
        LOGGER.debug("Effective learning rate: %.3e", total_gpu_count * self.config.training.lr.rate)
        LOGGER.debug("Rollout window length: %d", self.config.training.rollout.start)

    def update_paths(self) -> None:
        """Update the paths in the configuration."""
        self.config.hardware.paths.checkpoints = Path(self.config.hardware.paths.checkpoints, self.run_id)
        self.config.hardware.paths.plots = Path(self.config.hardware.paths.plots, self.run_id)

    def compile(self) -> None:
        # this doesn't work ATM (April 2), don't bother enabling it ...
        LOGGER.debug("torch.compiling the Lightning model ...")
        self.model = torch.compile(self.model, mode="default", backend="inductor", fullgraph=False)

    @cached_property
    def strategy(self) -> Any:
        return DDPGroupStrategy(self.config.hardware.group_size, static_graph=True)

    def train(self) -> None:
        """Training entry point."""

        if self.config.training.compile:
            self.compile()

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=self.callbacks,
            deterministic=self.config.training.deterministic,
            detect_anomaly=self.config.diagnostics.debug.anomaly_detection,
            strategy=self.strategy,
            devices=self.config.hardware.num_gpus_per_node,
            num_nodes=self.config.hardware.num_nodes,
            precision=self.config.training.precision,
            max_epochs=self.config.training.max_epochs,
            logger=self.loggers,
            log_every_n_steps=self.config.diagnostics.logging.interval,
            # run a fixed no of batches per epoch (helpful when debugging)
            limit_train_batches=self.config.dataloader.limit_batches.training,
            limit_val_batches=self.config.dataloader.limit_batches.validation,
            num_sanity_val_steps=0,
            accumulate_grad_batches=self.config.training.accum_grad_batches,
            gradient_clip_val=self.config.training.gradient_clip.val,
            gradient_clip_algorithm=self.config.training.gradient_clip.algorithm,
            # we have our own DDP-compliant sampler logic baked into the dataset
            use_distributed_sampler=False,
            profiler=self.profiler,
        )

        trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=self.last_checkpoint)
        LOGGER.debug("---- DONE. ----")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    AIFSTrainer(config).train()
