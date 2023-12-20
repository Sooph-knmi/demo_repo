import os
from pathlib import Path
from typing import Callable
from typing import List
from typing import Tuple

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose
from hydra import initialize
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric import seed_everything

from aifs.data.datamodule import ECMLDataModule
from aifs.distributed.helpers import gather_tensor
from aifs.losses.kcrps import KernelCRPS
from aifs.model.gnn import GraphMSG
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)
PRECISION = torch.float16


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


def build_avg_matrix_for_gather(
    ens_comm_group_size: int, model_comm_group_size: int, nens_per_device: int, rank: int
) -> torch.Tensor:
    """Builds a matrix of shape (ens_comm_group_size * nens_per_device, num_model_groups
    * nens_per_device). This matrix is used to average the contributions of individual
    ensemble members gathered in the ensemble comm group. It accounts for duplicates and
    different model sharding communication groups, if applicable.

    E.g., suppose
        - nens_per_device = 3
        - ens_comm_group_size = 4
        - model_comm_group_size = 2 (i.e. 2 model comm groups, and a total of 6 unique ensemble members)
    Then the gather matrix has shape (12, 6) and looks like:
        - * ( 0.5 * eye(3)  0.5 * eye(3)         0           0        )^T
        - * (      0              0        0.5 * eye(3)  0.5 * eye(3) )
    """
    num_model_groups = ens_comm_group_size // model_comm_group_size
    # sub-block used to average all contributions from a model comm group
    model_gather_mat = (1.0 / model_comm_group_size) * torch.cat(
        [torch.eye(nens_per_device, dtype=PRECISION, device=rank)] * model_comm_group_size, dim=1
    )
    ens_gather_mat = torch.block_diag(*([model_gather_mat] * num_model_groups)).T
    LOGGER.debug("Rank %d -- gather matrix shape = %s", rank, list(ens_gather_mat.shape))

    torch.set_printoptions(precision=2)
    LOGGER.debug("Rank %d -- gather matrix: \n%s", rank, ens_gather_mat)
    torch.set_printoptions(precision=8)

    return ens_gather_mat


def grad_output_hook(tname: str, rank: int) -> Callable:
    def _grad_output_hook(grad) -> torch.Tensor:
        if grad is not None:
            LOGGER.debug(
                "Rank %.1d: %s.grad.shape = %s, %s.grad.norm = %.5e", rank, tname, grad.shape, tname, torch.linalg.norm(grad)
            )
        else:
            LOGGER.error("Rank %.1d: %s.grad is None!", rank, tname)
        return grad

    return _grad_output_hook


def single_step_test(rank, world_size):
    LOGGER.debug("Runing GNN in DDP mode on rank %d ...", rank)
    setup(rank, world_size)

    initialize(config_path="../config", job_name="test_msg")
    cfg_ = compose(
        config_name="steptest",
        overrides=[
            "training.multistep_input=2",
            "model.trainable_parameters.era=2",
            "model.trainable_parameters.hidden=2",
            "model.trainable_parameters.era2hidden=2",
            "model.trainable_parameters.hidden2era=2",
            "model.trainable_parameters.hidden2hidden=2",
            "model.num_channels=32",
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
    torch.set_printoptions(precision=10)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    gather_mat = build_avg_matrix_for_gather(
        ens_comm_group_size, model_comm_group_size, cfg_.training.ensemble_size_per_device, rank
    )
    graph_data = torch.load(Path(cfg_.hardware.paths.graph, cfg_.hardware.files.graph))

    datamodule = ECMLDataModule(cfg_)
    data_indices = datamodule.data_indices

    # need to use torch.double for some communication operations
    # otherwise the gradients won't match up to the desired tolerances
    gnn = GraphMSG(config=cfg_, data_indices=data_indices, graph_data=graph_data).to(device=rank)
    _ERA_SIZE = 40320

    # DDP model wrapper
    gnn = DDP(gnn, device_ids=[rank])
    gnn = gnn.train()

    initial_params = get_parameters_as_a_flat_tensor(gnn)

    # random input tensor, shaped like what the model expects
    x_input = torch.randn(
        cfg_.dataloader.batch_size.training,
        cfg_.training.ensemble_size_per_device,
        cfg_.training.multistep_input,
        _ERA_SIZE,
        len(data_indices.data.input.full),
    ).to(rank)

    # loss in FP32
    kcrps = KernelCRPS(
        area_weights=torch.ones(_ERA_SIZE, dtype=torch.float32),
        loss_scaling=torch.ones(len(data_indices.model.output.full), dtype=torch.float32),
        fair=True,
    ).to(device=rank, dtype=torch.float32)
    kcrps = kcrps.train()

    y_target = torch.randn(  # already reshaped to fit the kcrps loss
        cfg_.dataloader.batch_size.training,
        len(data_indices.model.output.full),
        _ERA_SIZE,
        dtype=torch.float32,
        device=rank,
        requires_grad=False,
    )

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.SGD(gnn.parameters(), lr=1e-3, momentum=0.9)
    optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=PRECISION):
        y_pred = gnn(x_input, model_comm_group=model_comm_group, inject_noise=False)
        y_pred_ens = gather_tensor(y_pred, dim=1, shapes=[y_pred.shape] * ens_comm_group_size, mgroup=ens_comm_group)
        y_pred_ens = einops.rearrange(y_pred_ens, "bs e latlon v -> bs v latlon e")
        y_pred_ens = y_pred_ens @ gather_mat
        loss = kcrps(y_pred_ens, y_target, squash=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dist.barrier()

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
