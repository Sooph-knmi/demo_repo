import os
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose
from hydra import initialize
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric import seed_everything

from aifs.model.gnn import GraphMSG
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)


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


def get_parameters_as_a_flat_tensor(m: nn.Module) -> torch.Tensor:
    # save starting parameters
    p_flat: List[torch.Tensor] = []
    for p in m.parameters():
        if p.requires_grad:
            p_flat.append(p.view(-1).detach().clone())
    return torch.cat(p_flat)


def register_parameter_grad_scaling_hooks(m: nn.Module, grad_scaling_factor: float) -> None:
    """Register parameter hooks for gradient reduction.

    Here, we rescale parameters that only see a subset of the input on each rank
    -> these are still divided by the total number of GPUs in DDP as if
    each rank would see a full set of inputs
    """
    for _, param in m.named_parameters():
        if param.requires_grad:  #  and "trainable" not in name:
            param.register_hook(lambda grad: grad * grad_scaling_factor)


def single_step_test(rank, world_size):
    LOGGER.debug("Runing GNN in DDP mode on rank %d ...", rank)
    setup(rank, world_size)

    initialize(config_path="../config", job_name="test_msg")
    cfg_ = compose(
        config_name="steptest",
        overrides=[
            "data.num_features=8",
            "data.num_aux_features=2",
            "training.multistep_input=2",
            "model.trainable_parameters.era=2",
            "model.trainable_parameters.hidden=2",
            "model.trainable_parameters.era2hidden=2",
            "model.trainable_parameters.hidden2era=2",
            "model.trainable_parameters.hidden2hidden=2",
            "model.num_channels=8",
        ],
    )

    model_comm_group_size = cfg_.hardware.num_gpus_per_model
    ens_comm_group_size = cfg_.hardware.num_gpus_per_ensemble

    # create communication groups
    model_comm_group, _ = create_model_comm_groups(rank, world_size, model_comm_group_size)
    model_comm_group_size = dist.get_world_size(group=model_comm_group)
    LOGGER.debug("Set up a model_comm_group: %s with size %d", model_comm_group, model_comm_group_size)

    ens_comm_group, ens_comm_group_id = create_ensemble_comm_groups(rank, world_size, ens_comm_group_size)
    ens_comm_group_size = dist.get_world_size(group=ens_comm_group)
    LOGGER.debug("Set up an ensemble_comm_group: %s with size %d", ens_comm_group, ens_comm_group_size)

    iseed = 1234 + ens_comm_group_id
    seed_everything(iseed)
    LOGGER.debug("Rank %d has random seed %d", rank, iseed)
    torch.set_printoptions(precision=15)
    torch.use_deterministic_algorithms(True)

    graph_data = torch.load(Path(cfg_.hardware.paths.graph, cfg_.hardware.files.graph))

    gnn = GraphMSG(cfg_, graph_data).to(rank)
    _ERA_SIZE = gnn._era_size

    # DDP model wrapper
    gnn = DDP(gnn, device_ids=[rank])

    # TODO: is this the correct gradient scaling for all valid
    # (num-gpus-per-model, num-gpus-per-ensemble) combinations?
    register_parameter_grad_scaling_hooks(gnn, float(ens_comm_group_size))

    initial_params = get_parameters_as_a_flat_tensor(gnn)

    # random input tensor, shaped like what the model expects
    x_input = torch.randn(
        cfg_.dataloader.batch_size.training,
        cfg_.training.ensemble_size_per_device,
        cfg_.training.multistep_input,
        _ERA_SIZE,
        cfg_.data.num_features,
    ).to(rank)

    # y = torch.randn(
    #     cfg_.dataloader.batch_size.training, _ERA_SIZE, cfg_.data.num_features - cfg_.data.num_aux_features,
    # ).to(rank)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.SGD(gnn.parameters(), lr=1e-3, momentum=0.9)
    optimizer.zero_grad()
    working_precision = torch.float16

    with torch.autocast(device_type="cuda", dtype=working_precision):
        y_pred = gnn(x_input, model_comm_group=model_comm_group)
        # a dummy loss, for now
        loss = y_pred.sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        updated_params = get_parameters_as_a_flat_tensor(gnn)

        # calculate absolute & relative differences in the parameters (new vs old)
        LOGGER.debug("model sum(|p_new - p_init|): %.10e", torch.abs(updated_params - initial_params).sum())
        LOGGER.debug("model sum(abs(p)): %.10e", updated_params.abs().sum())

    LOGGER.debug("Rank %d max memory alloc: %.2f MB", rank, torch.cuda.max_memory_allocated(torch.device(rank)) / 1e6)
    LOGGER.debug("Rank %d max memory reserved: %.2f MB", rank, torch.cuda.max_memory_reserved(torch.device(rank)) / 1e6)

    cleanup()


def main():
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    LOGGER.debug("World size %d", world_size)
    run_parallel(single_step_test, world_size)


if __name__ == "__main__":
    main()
