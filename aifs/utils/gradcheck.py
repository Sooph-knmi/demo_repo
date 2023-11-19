from functools import cached_property
from typing import Any
from typing import List
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric import seed_everything

from aifs.data.era_datamodule import ERA5DataModule
from aifs.diagnostics.logging import get_wandb_logger
from aifs.distributed.strategy import DDPGroupStrategy
from aifs.train.forecaster import GraphForecaster
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class AIFSGradientChecker:
    """Utility class for checking model gradients."""

    def __init__(self, config: DictConfig):
        # Sets the internal precision of float32 matrix multiplications.
        torch.set_float32_matmul_precision("high")
        # Resolve the config to avoid shenanigans with lazy loading
        OmegaConf.resolve(config)
        self.config = config

        self.config.training.run_id = self.run_id

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
        return None

    @cached_property
    def loggers(self) -> List:
        loggers = []
        if self.config.diagnostics.log.wandb.enabled:
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
        total_number_of_model_instances = (
            self.config.hardware.num_nodes * self.config.hardware.num_gpus_per_node / self.config.hardware.num_gpus_per_model
        )
        LOGGER.debug(
            "Total GPU count / model group size: %d - NB: the learning rate will be scaled by this factor!",
            total_number_of_model_instances,
        )
        LOGGER.debug("Effective learning rate: %.3e", total_number_of_model_instances * self.config.training.lr.rate)
        LOGGER.debug("Rollout window length: %d", self.config.training.rollout.start)

    @cached_property
    def strategy(self) -> Any:
        return DDPGroupStrategy(
            self.config.hardware.num_gpus_per_model,
            self.config.hardware.num_gpus_per_ensemble,
            static_graph=self.config.training.accum_grad_batches > 1,
        )

    def _calculate_grad_norms(self) -> torch.Tensor:
        grad_norms_sum = torch.zeros(1, device=self.model.device, dtype=torch.double)
        for p in self.model.parameters():
            grad_norms_sum += torch.linalg.norm(p.grad)
        return grad_norms_sum

    def check_gradients(self) -> None:
        """Entry point."""
        seed_everything(1234)
        torch.use_deterministic_algorithms(True)

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=None,
            deterministic=True,  # self.config.training.deterministic,
            detect_anomaly=False,
            strategy=self.strategy,
            devices=self.config.hardware.num_gpus_per_node,
            num_nodes=self.config.hardware.num_nodes,
            # the grad checker requires double precision
            precision="64-true",
            max_epochs=1,
            logger=self.loggers,
            log_every_n_steps=1,
            # minimal warmup
            limit_train_batches=3,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=self.config.training.accum_grad_batches,
            # we have our own DDP-compliant sampler logic baked into the dataset
            use_distributed_sampler=False,
        )

        # run through 3 training batches, just to make sure we've set up everything correctly
        trainer.fit(self.model, datamodule=self.datamodule)

        # one manual forward-backward
        dl_train = self.datamodule.train_dataloader()

        # need to move the model back on to the GPU, it got offloaded at the end of trainer.fit()
        self.model = self.model.cuda()
        self.model = self.model.to(dtype=torch.double)
        self.model.train()
        torch.set_grad_enabled(True)

        # push a single batch through the model
        # run backward + check the gradient norms
        for batch_idx, batch in enumerate(dl_train):
            # LOGGER.debug("batch_idx: %03d, len(batch) = %d, batch[0].shape = %s", batch_idx, len(batch), batch[0].shape)
            # LOGGER.debug("model.device = %s", self.model.device)

            # move batch to correct device and convert to double
            batch = [b.to(device=self.model.device, dtype=torch.double) for b in batch]

            # push batch through the model
            x_ens_ic = self.model._generate_ens_inicond(batch)

            # LOGGER.debug("x_ens_ic: dtype = %s, device = %s", x_ens_ic.dtype, x_ens_ic.device)
            LOGGER.debug("Norm of ensemble ICs: %.9e", torch.linalg.norm(x_ens_ic))
            LOGGER.debug("Norm of batch data: %.9e", torch.linalg.norm(batch[0]))

            train_loss, _, _, _ = self.model._step(batch[0], batch_idx, x_ens_ic)
            train_loss.backward()
            LOGGER.debug("Ran backward ...")

            gnorm_sum = self._calculate_grad_norms()
            LOGGER.debug(
                "GlobalRank %d: batch_size=%d, gpus_per_model=%d, gpus_per_ensemble=%d, ens_per_device=%d - sum(||grad(p)||)=%.9e",
                self.model.global_rank,
                self.config.dataloader.batch_size.training,
                self.config.hardware.num_gpus_per_model,
                self.config.hardware.num_gpus_per_ensemble,
                self.config.training.ensemble_size_per_device,
                gnorm_sum,
            )

            # we're done
            break

        LOGGER.debug("---- GRADCHECK DONE. ----")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    AIFSGradientChecker(config).check_gradients()
