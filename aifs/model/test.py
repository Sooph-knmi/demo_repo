import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from xformers.components.attention import ScaledDotProduct

class simple_model(torch.nn.Module):
    def __init__(self, heads, in_channels, out_channels):
        super().__init__()

        self.heads=heads
        self.atten = ScaledDotProduct()
        self.lin_qkv = torch.nn.Linear(in_channels, 3*in_channels)

        self.proj = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):

        query, key, value = self.lin_qkv(x).chunk(3, -1)
        x = self.atten(query, key, value)

        return x

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12435"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_parallel(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

def test_model(rank, world_size):

    setup(rank, world_size)

    model = simple_model(16, 32, 32).to(rank)
    model_ddp = DDP(model, device_ids=[rank])
    # model_ddp = model

    x = torch.randn(2, 16, 2048, 32).to(rank)
    y = torch.randn(2, 16, 2048, 32).to(rank)

    y_pred = model_ddp(x)
    loss = (y_pred - y).sum()
    loss.backward()

    cleanup()

if __name__ == "__main__":
    import os

    n_gpus = torch.cuda.device_count()
    world_size = 1 #n_gpus
    print(f"world size {world_size}")
    run_parallel(test_model, world_size)
