from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.overrides.distributed import _sync_module_states
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)


class DDPGroupStrategy(DDPStrategy):
    def __init__(self, num_gpus_per_model: int, num_gpus_per_ensemble: int, **kwargs):
        super().__init__(**kwargs)
        self.model_comm_group_size = num_gpus_per_model
        self.ens_comm_group_size = num_gpus_per_ensemble

    def setup(self, trainer: pl.Trainer) -> None:
        assert self.accelerator is not None, "Accelerator is not initialized for distributed strategy"
        self.accelerator.setup(trainer)

        # determine the model groups that work together:
        assert self.world_size % self.model_comm_group_size == 0, (
            f"Total number of GPUs ({self.world_size}) must be divisible by the number of GPUs "
            f"per model ({self.model_comm_group_size})."
        )
        self._create_model_comm_groups()

        # determine the ensemble groups that work together:
        assert self.world_size % self.ens_comm_group_size == 0, (
            f"Total number of GPUs ({self.world_size}) must be divisible by the number of GPUs "
            f"per ensemble ({self.ens_comm_group_size})."
        )
        assert self.ens_comm_group_size % self.model_comm_group_size == 0, (
            f"Number of GPUs per ensemble ({self.ens_comm_group_size}) must be divisible by the number of GPUs "
            f"per model ({self.model_comm_group_size})."
        )
        self._create_ensemble_comm_groups()

        # register hooks for correct gradient reduction
        self.register_parameter_hooks()

        # move the model to the correct device
        self.model_to_device()

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING and self._layer_sync:
            assert self.model is not None, "Model is not initialized for distributed strategy"
            self.model = self._layer_sync.apply(self.model)

        self.setup_precision_plugin()

        if trainer_fn == TrainerFn.FITTING:
            # do not wrap with DDP if not fitting as there's no gradients to reduce
            self.configure_ddp()

            # set up optimizers after the wrapped module has been moved to the device
            self.setup_optimizers(trainer)
            _optimizers_to_device(self.optimizers, self.root_device)

            import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

            if isinstance(self._ddp_comm_state, post_localSGD.PostLocalSGDState):
                self._enable_model_averaging()
        else:
            # we need to manually synchronize the module's states since we aren't using the DDP wrapper
            assert self.model is not None, "Model is not initialized for distributed strategy"
            _sync_module_states(self.model)

    def _create_model_comm_groups(self):
        model_comm_group_ranks = np.split(np.arange(self.world_size, dtype=int), int(self.world_size / self.model_comm_group_size))
        LOGGER.debug(
            "world_size: %d, ens_comm_group_size: %d, model_comm_group_ranks: %s",
            self.world_size,
            self.model_comm_group_size,
            model_comm_group_ranks,
        )

        model_comm_groups = [
            torch.distributed.new_group(x) for x in model_comm_group_ranks
        ]  # every rank has to create all of these

        model_comm_group_id, model_comm_group_nr, model_comm_group_rank = self.get_my_comm_group(self.model_comm_group_size)
        model_comm_group = model_comm_groups[model_comm_group_id]
        self.model.set_model_comm_group(model_comm_group)
        LOGGER.debug(
            "Rank %d model_comm_group is %s, group number %d, with local group rank %d and comms_group_ranks %s",
            self.global_rank,
            str(model_comm_group_nr),
            model_comm_group_id,
            model_comm_group_rank,
            str(model_comm_group_ranks[model_comm_group_id]),
        )

    def _create_ensemble_comm_groups(self):
        ens_comm_group_ranks = np.split(np.arange(self.world_size, dtype=int), int(self.world_size / self.ens_comm_group_size))
        LOGGER.debug(
            "world_size: %d, ens_comm_group_size: %d, ens_comm_group_ranks: %s",
            self.world_size,
            self.ens_comm_group_size,
            ens_comm_group_ranks,
        )

        ens_comm_groups = [torch.distributed.new_group(x) for x in ens_comm_group_ranks]  # every rank has to create all of these

        ens_comm_group_id, ens_comm_group_nr, ens_comm_group_rank = self.get_my_comm_group(self.ens_comm_group_size)
        ens_comm_group = ens_comm_groups[ens_comm_group_id]
        self.model.set_ensemble_comm_group(ens_comm_group)
        LOGGER.debug(
            "Rank %d ensemble_comm_group is %s, group number %d, with local group rank %d and comms_group_ranks %s",
            self.global_rank,
            str(ens_comm_group_nr),
            ens_comm_group_id,
            ens_comm_group_rank,
            str(ens_comm_group_ranks[ens_comm_group_id]),
        )

    def get_my_comm_group(self, num_gpus_per_group) -> Tuple[int, int, int]:
        """Determine tasks that work together and from a model group."""
        comm_groups = np.arange(0, self.world_size, dtype=np.int32)
        comm_groups = np.split(comm_groups, self.world_size / num_gpus_per_group)

        comm_group_id = None
        for i, comm_group in enumerate(comm_groups):
            if self.global_rank in comm_group:
                comm_group_id = i
                comm_group_nr = comm_group
                comm_group_rank = np.ravel(np.asarray(comm_group == self.global_rank).nonzero())[0]
        return comm_group_id, comm_group_nr, comm_group_rank

    def register_parameter_hooks(self):
        """Register parameter hooks for gradient reduction.

        Here, we rescale parameters that only see a subset of the input on each rank
        -> these are still divided by the total number of GPUs in DDP as if each rank would see a full set of inputs
        note: the trainable parameters are added before the split across GPUs and are therefore not rescaled.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad is True and "trainable" not in name:
                param.register_hook(lambda grad: grad * float(self.model_comm_group_size))
