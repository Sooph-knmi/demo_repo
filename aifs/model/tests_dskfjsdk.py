from typing import List
from typing import Optional
from typing import Tuple

import einops
import numpy as np
import torch

import xformers.ops as xops
import xformers.components.attention as xatten
from xformers.components.attention.core import scaled_dot_product_attention

class simple_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.heads=16

        # self.atten = xops.memory_efficient_attention
        self.atten = xatten.ScaledDotProduct() #causal=is_causal, dropout=self.dropout,)
        # self.atten = scaled_dot_product_attention

        self.lin_qkv = torch.nn.Linear(in_channels, 3*in_channels)

        self.proj = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):

        B, N, C = x.shape

        query, key, value = self.lin_qkv(x).chunk(3, -1)

        # memory efficient attention
        # query, key, value = map(
        #     lambda t: einops.rearrange(t, "b n (h c) -> b n h c", b=B, h=self.heads), (query, key, value)
        # )
        # x = self.atten(query, key, value)
        # x = einops.rearrange(x, "b n h c -> b n (h c)")


        # ScaledDotProduct
        query, key, value = map(
            lambda t: einops.rearrange(t, "b n (h c) -> b h n c", b=B, h=self.heads), (query, key, value)
        )
        x = self.atten(query, key, value)
        x = einops.rearrange(x, "b h n c -> b n (h c)")

        x = self.proj(x)

        return x


############ testing ########


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_my_mgroup(world_size, rank, mgroup_size):
    """Determine model group."""
    mgroups = np.array([x for x in np.arange(0, world_size)])
    mgroups = np.split(mgroups, world_size / mgroup_size)

    my_mgroup = None
    imgroup = None
    for i, mgroup in enumerate(mgroups):
        if rank in mgroup:
            imgroup = i
            my_mgroup = mgroup
            mgroup_rank = np.ravel(np.asarray(mgroup == rank).nonzero())[0]
    return imgroup, my_mgroup, mgroup_rank


from torch_geometric import seed_everything

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from pathlib import Path

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  #'localhost'
    os.environ["MASTER_PORT"] = "12435"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_parallel(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def test_model(rank, world_size):

    setup(rank, world_size)

    model = simple_model(512, 512).to(rank)
    model_ddp = DDP(model, device_ids=[rank])

    optimizer = torch.optim.SGD(model_ddp.parameters(), lr=0.001, momentum=0.9)

    x = torch.randn(2, 2048, 512).to(rank)
    y = torch.randn(2, 2048, 512).to(rank)

    # scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()

    y_pred = model_ddp(x)
    loss = (y_pred - y).sum()
    loss.backward()

    # with torch.autocast(device_type="cuda", dtype=torch.float16):
    #     y_pred = model_ddp(x)
    #     loss = (y_pred - y).sum()

    # if True:
    #     scaler.scale(loss).backward()

    #     print("{rank} --- %s seconds ---" % (time.time() - start_time))

    #     print(f" =====##### rank {rank} has loss1 {loss:.20f}")

    #     scaler.step(optimizer)
    #     scaler.update()

    print(f"rank {rank} max memory alloc: {torch.cuda.max_memory_allocated(torch.device(rank))/1.e6}")
    print(f"rank {rank} max memory reserved: {torch.cuda.max_memory_reserved(torch.device(rank))/1.e6}")

    print("done")
    cleanup()


if __name__ == "__main__":
    import os

    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    print(f"world size {world_size}")
    run_parallel(test_model, world_size)
