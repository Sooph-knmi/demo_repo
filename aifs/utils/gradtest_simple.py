import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from hydra import compose
from hydra import initialize
from torch.autograd import gradcheck
from torch.distributed import destroy_process_group
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric import seed_everything

from aifs.model.gnn import GraphMSG
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def gradient_test(rank: int, world_size: int):
    print(f"Running DDP with model parallel example on rank {rank}.")
    ddp_setup(rank, world_size)
    LOGGER.debug("DDP setup -- world size: %d -- rank : %d", world_size, rank)

    initialize(config_path="../config", job_name="test_msg")
    cfg_ = compose(
        config_name="gradcheck",
        overrides=[
            'hardware.files.graph="graph_mappings_normed_edge_attrs_rot_o96_h3_0_1_2.pt"',
            "data.num_features=8",
            "data.num_aux_features=2",
            "training.multistep_input=1",
            "model.trainable_parameters.era=2",
            "model.trainable_parameters.hidden=2",
            "model.trainable_parameters.era2hidden=2",
            "model.trainable_parameters.hidden2era=2",
            "model.trainable_parameters.hidden2hidden=2",
            "model.num_channels=8",
        ],
    )

    gdata = torch.load(Path(cfg_.hardware.paths.graph, cfg_.hardware.files.graph))
    gnn_ = GraphMSG(cfg_, graph_data=gdata).to(rank, dtype=torch.float64)
    _ERA_SIZE = gnn_._era_size

    # wrap model
    gnn_ = DDP(gnn_, device_ids=[rank])

    # x_input = torch.randn(
    #     cfg_.dataloader.batch_size.training,
    #     cfg_.training.ensemble_size_per_device,
    #     cfg_.training.multistep_input,
    #     _ERA_SIZE,
    #     cfg_.data.num_features,
    #     # must (1) create it directly on the correct device and (2) set the requires_grad flag
    #     dtype=torch.float32,
    #     device=rank,
    #     requires_grad=True
    # )

    # LOGGER.debug("Input shape: %s", x_input.shape)

    # y_pred = gnn_(x_input)
    # LOGGER.debug("Output shape: %s", y_pred.shape)

    # loss = y_pred.sum()
    # LOGGER.debug("Running backward on a simple sum loss ...")
    # loss.backward()
    # LOGGER.debug("Ran backward. All good!")

    # LOGGER.debug("|| x_input.grad || = %.10e", torch.linalg.norm(x_input.grad))

    # barrier()

    real_input_shape = (
        cfg_.dataloader.batch_size.training,
        cfg_.training.ensemble_size_per_device,
        cfg_.training.multistep_input,
        _ERA_SIZE,
        cfg_.data.num_features,
    )

    fake_input_shape = (10,)

    linear_map = torch.nn.Linear(np.prod(fake_input_shape), np.prod(real_input_shape), bias=False, device=rank, dtype=torch.float64)

    def gnn_with_dummy_loss(x_inp: torch.Tensor) -> torch.Tensor:
        x_out = gnn_(linear_map(x_inp).reshape(real_input_shape))
        return x_out.sum()

    # x_test = torch.randn(
    #     cfg_.dataloader.batch_size.training,
    #     cfg_.training.ensemble_size_per_device,
    #     cfg_.training.multistep_input,
    #     _ERA_SIZE,
    #     cfg_.data.num_features,
    #     # must (1) create it directly on the correct device and (2) set the requires_grad flag
    #     dtype=torch.float64,  # needs double precision
    #     device=rank,
    #     requires_grad=True
    # )

    # low-dimensional input. it'll get mapped to the right shape by linear_map above
    x_test = torch.randn(
        fake_input_shape,
        # must (1) create it directly on the correct device and (2) set the requires_grad flag
        dtype=torch.float64,  # needs double precision
        device=rank,
        requires_grad=True,
    )

    test = gradcheck(gnn_with_dummy_loss, (x_test,), eps=1e-6, atol=1e-4, rtol=1e-2, nondet_tol=0.0)
    LOGGER.debug(test)

    destroy_process_group()


if __name__ == "__main__":
    seed_everything(1234)
    torch.use_deterministic_algorithms(True)

    # TODO: this will work on a single node only
    ntasks = int(os.getenv("SLURM_PROCID", "0"))
    world_size = ntasks if ntasks > 0 else torch.cuda.device_count()
    rank = int(os.getenv("SLURM_PROCID", "0"))
    LOGGER.debug("World size: %d -- rank : %d", world_size, rank)
    assert world_size >= 1, f"Requires at least 1 GPU to run, but got {world_size}!"
    mp.spawn(gradient_test, args=(world_size,), nprocs=world_size)
