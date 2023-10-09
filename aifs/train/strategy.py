import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.overrides.distributed import _sync_module_states
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


class DDPGroupStrategy(DDPStrategy):
    def __init__(self, num_gpus_per_model: int, **kwargs):
        super().__init__(**kwargs)
        self.num_gpus_per_model = num_gpus_per_model

    def setup(self, trainer: pl.Trainer) -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        # determine the model groups that work together:

        assert self.world_size % self.num_gpus_per_model == 0

        comms_groups_ranks = np.split(np.array([x for x in range(0, self.world_size)]), int(self.world_size / self.num_gpus_per_model))
        comms_groups = [torch.distributed.new_group(x) for x in comms_groups_ranks] # every rank has to create all of these

        id_model_coms_group, my_model_coms_group, my_model_coms_group_rank = self.get_my_model_coms_group(self.num_gpus_per_model)
        comms_group = comms_groups[id_model_coms_group]
        self.model.set_model_coms_groups(comms_group)
        LOGGER.debug(
            "Rank %d model_coms_group is %s, group number %d, with local group rank %d and comms_group_ranks %s",
            self.global_rank,
            str(my_model_coms_group),
            id_model_coms_group,
            my_model_coms_group_rank,
            str(comms_groups_ranks[id_model_coms_group]),
        )

        # register hooks for correct gradient reduction
        self.register_parameter_hooks()

        # move the model to the correct device
        self.model_to_device()

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING and self._layer_sync:
            assert self.model is not None
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
            assert self.model is not None
            _sync_module_states(self.model)

    def get_my_model_coms_group(self, num_gpus_per_model):
        """Determine tasks that work together and from a model group."""
        model_coms_groups = np.arange(0, self.world_size, dtype=np.int32)
        model_coms_groups = np.split(model_coms_groups, self.world_size / num_gpus_per_model)

        my_model_coms_group = None
        id_model_coms_group = None
        for i, model_coms_group in enumerate(model_coms_groups):
            if self.global_rank in model_coms_group:
                id_model_coms_group = i
                my_model_coms_group = model_coms_group
                model_coms_group_rank = np.ravel(np.asarray(model_coms_group == self.global_rank).nonzero())[0]
        return id_model_coms_group, my_model_coms_group, model_coms_group_rank

    def register_parameter_hooks(self):
        """Register parameter hooks for gradient reduction.
        Here, we rescale parameters that only see a subset of the input on each rank 
        -> these are still divided by the total number of GPUs in DDP as if each rank would see a full set of inputs
        note: the trainable parameters are added before the split across GPUs and are therefore not rescaled.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad == True and "trainable" not in name:
                param.register_hook(lambda grad: grad * float(self.num_gpus_per_model))
