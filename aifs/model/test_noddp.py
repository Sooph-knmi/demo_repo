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
        self.atten = xatten.ScaledDotProduct()
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


from torch_geometric import seed_everything

if __name__ == "__main__":

    iseed = 1234
    seed_everything(iseed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # required
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8

    torch.set_printoptions(precision=25)

    iseed = 1234
    cuda0 = torch.device(0)
    gnn = simple_model(512, 512).to(cuda0)

    optimizer = torch.optim.SGD(gnn.parameters(), lr=0.001, momentum=0.9)

    ### data:
    x = torch.randn(2, 2048, 512).to(cuda0)
    y = torch.randn(2, 2048, 512).to(cuda0)

    scaler = torch.cuda.amp.GradScaler()

    optimizer.zero_grad()

    # start_time = time.time()

    cast_to = torch.float16

    with torch.autocast(device_type="cuda", dtype=cast_to):
        y_pred = gnn(x)
        loss = (y_pred - y).sum()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(loss)
    # print(f"rank {rank} max memory alloc: {torch.cuda.max_memory_allocated(torch.device(rank))/1.e6}")
    # print(f"rank {rank} max memory reserved: {torch.cuda.max_memory_reserved(torch.device(rank))/1.e6}")

    print("done")
