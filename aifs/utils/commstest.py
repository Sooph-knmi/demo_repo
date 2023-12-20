import os
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose
from hydra import initialize
from torch import nn
from torch.autograd import gradcheck
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric import seed_everything

from aifs.distributed.helpers import gather_tensor
from aifs.distributed.helpers import reduce_shard_tensor
from aifs.distributed.helpers import reduce_tensor
from aifs.distributed.helpers import shard_tensor
from aifs.distributed.helpers import sync_tensor
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)

_TEST_INPUT_SHAPE = (5,)
_TEST_DTYPE = torch.float64


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12435"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_parallel(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def create_model_comm_groups(global_rank, world_size, model_comm_group_size):
    model_comm_group_ranks = np.split(np.arange(world_size, dtype=int), int(world_size / model_comm_group_size))
    LOGGER.debug(
        "world_size: %d, ens_comm_group_size: %d, model_comm_group_ranks: %s",
        world_size,
        model_comm_group_size,
        model_comm_group_ranks,
    )

    model_comm_groups = [torch.distributed.new_group(x) for x in model_comm_group_ranks]  # every rank has to create all of these

    model_comm_group_id, model_comm_group_nr, model_comm_group_rank = get_my_comm_group(
        global_rank, world_size, model_comm_group_size
    )
    model_comm_group = model_comm_groups[model_comm_group_id]
    LOGGER.debug(
        "Rank %d model_comm_group is %s, group number %d, with local group rank %d and comms_group_ranks %s",
        global_rank,
        str(model_comm_group_nr),
        model_comm_group_id,
        model_comm_group_rank,
        str(model_comm_group_ranks[model_comm_group_id]),
    )
    return model_comm_group, model_comm_group_id


def create_ensemble_comm_groups(global_rank, world_size, ens_comm_group_size):
    ens_comm_group_ranks = np.split(np.arange(world_size, dtype=int), int(world_size / ens_comm_group_size))
    LOGGER.debug(
        "world_size: %d, ens_comm_group_size: %d, ens_comm_group_ranks: %s",
        world_size,
        ens_comm_group_size,
        ens_comm_group_ranks,
    )

    ens_comm_groups = [torch.distributed.new_group(x) for x in ens_comm_group_ranks]  # every rank has to create all of these
    ens_comm_group_id, ens_comm_group_nr, ens_comm_group_rank = get_my_comm_group(global_rank, world_size, ens_comm_group_size)
    ens_comm_group = ens_comm_groups[ens_comm_group_id]

    LOGGER.debug(
        "Rank %d ensemble_comm_group is %s, group number %d, with local group rank %d and comms_group_ranks %s",
        global_rank,
        str(ens_comm_group_nr),
        ens_comm_group_id,
        ens_comm_group_rank,
        str(ens_comm_group_ranks[ens_comm_group_id]),
    )

    return ens_comm_group, ens_comm_group_id


def get_my_comm_group(global_rank, world_size, num_gpus_per_group) -> Tuple[int, int, int]:
    """Determine tasks that work together and from a model group."""
    comm_groups = np.arange(0, world_size, dtype=np.int32)
    comm_groups = np.split(comm_groups, world_size / num_gpus_per_group)

    comm_group_id = None
    for i, comm_group in enumerate(comm_groups):
        if global_rank in comm_group:
            comm_group_id = i
            comm_group_nr = comm_group
            comm_group_rank = np.ravel(np.asarray(comm_group == global_rank).nonzero())[0]
    return comm_group_id, comm_group_nr, comm_group_rank


def register_parameter_grad_scaling_hooks(m: nn.Module, grad_scaling_factor: float) -> None:
    """Register parameter hooks for gradient reduction.

    Here, we rescale parameters that only see a subset of the input on each rank
    -> these are still divided by the total number of GPUs in DDP as if
    each rank would see a full set of inputs
    """
    for _, param in m.named_parameters():
        if param.requires_grad:  #  and "trainable" not in name:
            param.register_hook(lambda grad: grad * grad_scaling_factor)


def test_gather_op(comm_group: ProcessGroup, input_shape: Tuple, rank: int) -> bool:
    """Gradient test for gather_tensor collective.

    Inputs:
        comm_group: ProcessGroup
            Process communication group.
        input_shape: Tuple
            Shape of input to the communication operation.
        rank: int
            Global rank of current process.
    Outputs:
        True if the test passed OK.
    """
    small_input_shape = _TEST_INPUT_SHAPE
    comm_group_size = comm_group.size()
    x_input = torch.randn(*small_input_shape, dtype=_TEST_DTYPE, device=rank, requires_grad=True)

    linear_map = torch.nn.Linear(
        np.prod(small_input_shape),
        np.prod(input_shape),
        bias=False,
        device=rank,
        dtype=_TEST_DTYPE,
    )
    linear_map = linear_map.train()

    def _test_gather(x_in) -> None:
        x = linear_map(x_in).reshape(input_shape)
        y = gather_tensor(x, dim=1, shapes=[x.shape] * comm_group_size, mgroup=comm_group)
        loss = y.sum()
        return loss

    result = gradcheck(_test_gather, (x_input,), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0)
    LOGGER.debug("Gather test result: %s", result)  # "True" if the test passed
    return result


def test_shard_op(comm_group: ProcessGroup, input_shape: Tuple, rank: int) -> bool:
    """Gradient test for shard_tensor collective.

    Inputs:
        comm_group: ProcessGroup
            Process communication group.
        input_shape: Tuple
            Shape of input to the communication operation.
        rank: int
            Global rank of current process.
    Outputs:
        True if the test passed OK.
    """
    small_input_shape = _TEST_INPUT_SHAPE
    comm_group_size = comm_group.size()
    x_input = torch.randn(*small_input_shape, dtype=_TEST_DTYPE, device=rank, requires_grad=True)

    linear_map = torch.nn.Linear(
        np.prod(small_input_shape),
        np.prod(input_shape),
        bias=False,
        device=rank,
        dtype=_TEST_DTYPE,
    )
    linear_map = linear_map.train()

    def _test_shard(x_in) -> None:
        x_ = linear_map(x_in).reshape(input_shape)
        x = x_.repeat(1, comm_group_size, 1, 1)
        y = shard_tensor(x, dim=1, shapes=[x_.shape] * comm_group_size, mgroup=comm_group)
        loss = y.sum()
        return loss

    result = gradcheck(_test_shard, (x_input,), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0)
    LOGGER.debug("Shard test result: %s", result)  # "True" if the test passed
    return result


def test_sync_op(comm_group: ProcessGroup, input_shape: Tuple, rank: int) -> bool:
    """Gradient test for sync_tensor collective.

    Inputs:
        comm_group: ProcessGroup
            Process communication group.
        input_shape: Tuple
            Shape of input to the communication operation.
        rank: int
            Global rank of current process.
    Outputs:
        True if the test passed OK.
    """
    small_input_shape = _TEST_INPUT_SHAPE
    comm_group_size = comm_group.size()
    x_input = torch.randn(*small_input_shape, dtype=_TEST_DTYPE, device=rank, requires_grad=True)

    linear_map = torch.nn.Linear(
        np.prod(small_input_shape),
        np.prod(input_shape),
        bias=False,
        device=rank,
        dtype=_TEST_DTYPE,
    )
    linear_map = linear_map.train()

    def _test_sync(x_in) -> None:
        x = linear_map(x_in).reshape(input_shape)
        y = sync_tensor(x, dim=1, shapes=[x.shape] * comm_group_size, mgroup=comm_group)
        loss = y.sum()
        return loss

    result = gradcheck(_test_sync, (x_input,), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0)
    LOGGER.debug("Sync test result: %s", result)  # "True" if the test passed
    return result


def test_reduce_shard_op(comm_group: ProcessGroup, input_shape: Tuple, rank: int) -> bool:
    """Gradient test for reduce_shard_tensor collective.

    Inputs:
        comm_group: ProcessGroup
            Process communication group.
        input_shape: Tuple
            Shape of input to the communication operation.
        rank: int
            Global rank of current process.
    Outputs:
        True if the test passed OK.
    """
    small_input_shape = _TEST_INPUT_SHAPE
    comm_group_size = comm_group.size()
    x_input = torch.randn(*small_input_shape, dtype=_TEST_DTYPE, device=rank, requires_grad=True)

    linear_map = torch.nn.Linear(
        np.prod(small_input_shape),
        np.prod(input_shape),
        bias=False,
        device=rank,
        dtype=_TEST_DTYPE,
    )
    linear_map = linear_map.train()

    def _test_reduce_shard(x_in) -> None:
        x_ = linear_map(x_in).reshape(input_shape)
        x = x_.repeat(1, comm_group_size, 1, 1)
        # must do the reduction in FP64 (= _TEST_DTYPE) otherwise the gradient test will break
        y = reduce_shard_tensor(x, dim=1, shapes=[x_.shape] * comm_group_size, mgroup=comm_group, use_fp32=False)
        loss = y.sum()
        return loss

    result = gradcheck(_test_reduce_shard, (x_input,), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=0.0)
    LOGGER.debug("Reduce_shard test result: %s", result)  # "True" if the test passed
    return result


def test_reduce_op(comm_group: ProcessGroup, input_shape: Tuple, rank: int) -> bool:
    """Gradient test for reduce_tensor collective.

    Inputs:
        comm_group: ProcessGroup
            Process communication group.
        input_shape: Tuple
            Shape of input to the communication operation.
        rank: int
            Global rank of current process.
    Outputs:
        True if the test passed OK.
    """
    small_input_shape = _TEST_INPUT_SHAPE
    x_input = torch.randn(*small_input_shape, dtype=_TEST_DTYPE, device=rank, requires_grad=True)

    linear_map = torch.nn.Linear(
        np.prod(small_input_shape),
        np.prod(input_shape),
        bias=False,
        device=rank,
        dtype=_TEST_DTYPE,
    )
    linear_map = linear_map.train()

    def _test_reduce(x_in) -> None:
        x = linear_map(x_in).reshape(input_shape)
        # must do the reduction in FP64 (= _TEST_DTYPE) otherwise the gradient test will break
        y = reduce_tensor(x, mgroup=comm_group, use_fp32=False)
        loss = y.sum()
        return loss

    result = gradcheck(_test_reduce, (x_input,), eps=1e-7, atol=1e-5, rtol=1e-4, nondet_tol=0.0)
    LOGGER.debug("Reduce test result: %s", result)  # "True" if the test passed
    return result


def comms_test(rank, world_size):
    LOGGER.debug("Runing GNN in DDP mode on rank %d ...", rank)
    setup(rank, world_size)

    initialize(config_path="../config", job_name="test_msg")
    cfg_ = compose(
        config_name="commstest",
        overrides=[
            "training.multistep_input=2",
            "training.ensemble_size_per_device=3",
            "model.trainable_parameters.era=2",
            "model.trainable_parameters.hidden=2",
            "model.trainable_parameters.era2hidden=2",
            "model.trainable_parameters.hidden2era=2",
            "model.trainable_parameters.hidden2hidden=2",
            "model.num_channels=8",
        ],
    )

    # model_comm_group_size = cfg_.hardware.num_gpus_per_model
    ens_comm_group_size = cfg_.hardware.num_gpus_per_ensemble

    ens_comm_group, ens_comm_group_id = create_ensemble_comm_groups(rank, world_size, ens_comm_group_size)
    ens_comm_group_size = dist.get_world_size(group=ens_comm_group)
    LOGGER.debug("Set up an ensemble_comm_group: %s with size %d", ens_comm_group, ens_comm_group_size)

    iseed = 1234 + ens_comm_group_id
    seed_everything(iseed)
    LOGGER.debug("Rank %d has random seed %d", rank, iseed)
    torch.set_printoptions(precision=6)
    torch.use_deterministic_algorithms(True)

    _ERA_SIZE = 2048
    _NUM_FEATURES = 8

    # test gather
    input_shape = (
        cfg_.dataloader.batch_size.training,
        cfg_.training.ensemble_size_per_device,
        _ERA_SIZE,
        _NUM_FEATURES,
    )

    # test gather
    assert test_gather_op(ens_comm_group, input_shape, rank)

    # test sharding
    assert test_shard_op(ens_comm_group, input_shape, rank)

    # test sync
    assert test_sync_op(ens_comm_group, input_shape, rank)

    # test reduce + shard
    assert test_reduce_shard_op(ens_comm_group, input_shape, rank)

    # test reduce
    assert test_reduce_op(ens_comm_group, input_shape, rank)

    cleanup()


def main():
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    LOGGER.debug("World size %d", world_size)
    run_parallel(comms_test, world_size)


if __name__ == "__main__":
    main()
