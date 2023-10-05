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
    def __init__(self, group_size: int, **kwargs):
        super().__init__(**kwargs)
        self.mgroup_size = group_size

    def setup(self, trainer: pl.Trainer) -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        # determine the model groups that work together:

        assert self.world_size % self.mgroup_size == 0

        world_size = self.world_size
        my_rank = self.global_rank

        comms_groups_ranks = np.split(np.array([x for x in range(0, self.world_size)]), int(self.world_size / self.mgroup_size))
        comms_groups = [torch.distributed.new_group(x) for x in comms_groups_ranks]
        comms_groups_single = [torch.distributed.new_group((x,)) for x in range(self.world_size)]

        imgroup, my_mgroup, my_mgroup_rank = self.get_my_mgroup(world_size, my_rank, self.mgroup_size)
        comms_group = comms_groups[imgroup]
        self.model.set_mgroups(comms_group, comms_groups_single[my_rank])
        LOGGER.debug(
            "Rank %d mgroup is %s, group number %d, with local group rank %d and comms_group_ranks %s",
            my_rank,
            str(my_mgroup),
            imgroup,
            my_mgroup_rank,
            str(comms_groups_ranks[imgroup]),
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

    def get_my_mgroup(self, world_size, rank, mgroup_size):
        """Determine model group."""
        mgroups = np.arange(0, world_size, dtype=np.int32)
        mgroups = np.split(mgroups, world_size / mgroup_size)

        my_mgroup = None
        imgroup = None
        for i, mgroup in enumerate(mgroups):
            if rank in mgroup:
                imgroup = i
                my_mgroup = mgroup
                mgroup_rank = np.ravel(np.asarray(mgroup == rank).nonzero())[0]
        return imgroup, my_mgroup, mgroup_rank

    def register_parameter_hooks(self):
                for name, param in self.model.named_parameters():
                    if not "trainable" in name:
                        if param.requires_grad == True:
                            param.register_hook(lambda grad: grad * float(self.mgroup_size))
