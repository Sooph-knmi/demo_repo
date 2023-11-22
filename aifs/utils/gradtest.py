from functools import cached_property
from typing import Any
from typing import List
from typing import Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.autograd import gradcheck
from torch_geometric import seed_everything

from aifs.data.era_datamodule import ERA5DataModule
from aifs.diagnostics.logging import get_wandb_logger
from aifs.distributed.strategy import DDPGroupStrategy
from aifs.train.forecaster import GraphForecaster
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class AIFSGradientTester:
    r"""
    Utility class for testing model gradients: autodiff vs finite-difference Jacobians.
    To be used together with gradtest.yaml. You can try to run small, single-node configurations e.g.:
    (num_gpus_per_model, num_gpus_per_ensemble) \in { (1, 1), (1, 2), (1, 4), (2, 4), (2, 2), (4, 4) }
    The gradient test should complete without error in all these cases.
    """

    def __init__(self, config: DictConfig):
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

    def test_gradients(self) -> None:
        """Entry point."""
        seed_everything(1234 + self.model.ens_comm_group_id)  # different seeds for different ensemble groups
        torch.use_deterministic_algorithms(True)

        trainer = pl.Trainer(
            accelerator=self.accelerator,
            callbacks=None,
            deterministic=True,  # self.config.training.deterministic,
            detect_anomaly=True,
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

        self.model.kcrps.weights = self.model.kcrps.weights.to(dtype=torch.double, device=self.model.device)
        #### THE `scale` TENSOR BREAKS THE GRADIENT TEST - so we replace it by all ones
        self.model.kcrps.scale = torch.ones_like(self.model.kcrps.scale, dtype=torch.double, device=self.model.device)

        batch = next(iter(dl_train))[0]  # a single tensor is all we need
        batch_shape = batch.shape

        # A trick: I use a low-dimensional input, so gradcheck doesn't take ages to run
        # If the gradients check out okay for loss(model(x)), they will be for the
        # composite mapping loss(model(linear_map(x_small))), too (and vice versa)
        # the input will get mapped to the original batch shape by the linear_map
        fake_input_shape = (5,)
        linear_map = torch.nn.Linear(
            np.prod(fake_input_shape), np.prod(batch.shape), bias=False, device=self.model.device, dtype=torch.double
        )
        linear_map = linear_map.train()

        # TODO: simplify all this - simply set the scale tensor to torch.ones() inside the model (?)
        def single_forward(fake_batch) -> torch.Tensor:
            batch_ = linear_map(fake_batch).reshape(batch_shape)
            x_ = batch_[:, 0 : self.model.multi_step, ...]
            x_ = torch.stack([x_] * self.model.nens_per_device, dim=1)

            batch_ = self.model.model.normalizer(batch_, in_place=False)
            x_ = self.model.model.normalizer(x_, in_place=False)

            # single rollout step
            y_pred = self.model(x_)
            # need to scale the gradient of y_pred here (better to do this in the backward of the comm op?)
            y_pred.register_hook(lambda grad: grad * self.model.ens_comm_group_size)

            y = batch_[:, self.model.multi_step, :, : self.model.fcdim]

            # simple L1-like loss
            # loss_ = torch.abs(y_pred - y[:, None, ...]).sum()

            # y_pred_ens_ = gather_tensor(
            #     y_pred, dim=1, shapes=[y_pred.shape] * self.model.ens_comm_group_size, mgroup=self.model.ens_comm_group
            # )

            # y_pred_ens = y_pred_ens_[:, :: self.model.model_comm_group_size, ...]

            # # simple L1-like loss
            # loss_ = torch.abs(y_pred_ens - y[:, None, ...]).sum()

            _, loss_, _ = self.model.gather_and_compute_loss(y_pred, y)

            return loss_

        x_test = torch.randn(
            fake_input_shape,
            # must (1) create it directly on the correct device and (2) set the requires_grad flag
            dtype=torch.double,  # needs double precision
            device=self.model.device,
            requires_grad=True,
        )

        # the finite-diff vs analytical Jacobian check happens here
        test_result = gradcheck(single_forward, (x_test,), eps=1e-6, atol=1e-4, rtol=1e-2, nondet_tol=0.0)
        LOGGER.debug(test_result)  # "True" if the test passed

        LOGGER.debug("---- GRAD TEST DONE. ----")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    AIFSGradientTester(config).test_gradients()
