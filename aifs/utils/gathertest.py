import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric import seed_everything

from aifs.distributed.helpers import gather_tensor
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)

_TEST_DTYPE = torch.int64


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12435"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_parallel(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def gather_test(rank, world_size):
    LOGGER.debug("Runing GNN in DDP mode on rank %d ...", rank)
    setup(rank, world_size)

    seed_everything(1234 + rank)

    comm_group_ranks = np.arange(world_size, dtype=int)
    comm_group = torch.distributed.new_group(comm_group_ranks)  # every rank has to create this comm group

    iter_ = 0
    while iter_ < 100:
        x_input = torch.zeros(1, dtype=_TEST_DTYPE, device=rank) + rank
        x_gathered = gather_tensor(x_input, dim=0, shapes=[x_input.shape] * world_size, mgroup=comm_group)
        x_ref = torch.from_numpy(np.arange(world_size)).to(device=rank, dtype=_TEST_DTYPE)
        assert torch.allclose(x_gathered, x_ref), f"Expected {x_ref} and got {x_gathered}"
        iter_ = iter_ + 1
        dist.barrier()

    cleanup()


def main():
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    LOGGER.debug("World size %d", world_size)
    run_parallel(gather_test, world_size)


if __name__ == "__main__":
    main()
