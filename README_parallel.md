### [Lightning DDP](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html#distributed-data-parallel)

DDP works as follows:

1. Each GPU across each node gets its own process.
2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.
3. Each process inits the model.
4. Each process performs a full forward and backward pass in parallel.
5. The gradients are synced and averaged across all processes.
6. Each process updates its optimizer.

With Dask, it's important to make sure that only the "root" process sets up the cluster, the others just create their own clients.
Otherwise we end up with multiple Dask clusters. 

With DDP the model doesn't need to be pickled (so there are no pickle-related limitations).

**Q**: OK to have one Dask cluster per node? Would this be more efficient?

The Lightning implementation of DDP calls your script under the hood multiple times with the correct environment variables:

```shell
# example for 3 GPUs DDP
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=1 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=2 LOCAL_RANK=0 python my_file.py --accelerator 'gpu' --devices 3 --etc
```

DDP _cannot_ be used:

- in a Jupyter Notebook, Google COLAB, Kaggle, etc.
- in a nested script without a root package

DDP variants and tradeoffs: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html?highlight=DDP#comparison-of-ddp-variants-and-tradeoffs

#### DDP **optimizations**:

https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#ddp-optimizations


### Debugging

To debug a distributed model, debug it locally by running the distributed version on CPUs:

```python
trainer = Trainer(accelerator="cpu", strategy="ddp", devices=2)
```

On the CPU, you can use `pdb` or `breakpoint()` or use regular print statements.

```python
class LitModel(LightningModule):
    def training_step(self, batch, batch_idx):

        debugging_message = ...
        print(f"RANK - {self.trainer.global_rank}: {debugging_message}")

        if self.trainer.global_rank == 0:
            import pdb
            pdb.set_trace()

        # to prevent other processes from moving forward until all processes are in sync
        self.trainer.strategy.barrier()
```

### Batch size

When using distributed training make sure to modify your learning rate according to your effective batch size.
Let’s say you have a batch size of 7 in your dataloader.

```python
class LitModel(LightningModule):
    def train_dataloader(self):
        return Dataset(..., batch_size=7)
```

In `DDP`, `DDP_SPAWN`, `Deepspeed`, `DDP_SHARDED`, or Horovod your effective batch size will be `7 * devices * num_nodes`.

### [Lightning `global_rank`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#global-rank)

The `global_rank` is the index of the current process across all nodes and devices. 
Lightning will perform some operations such as logging, weight checkpointing only when `global_rank=0`. 
You usually do not need to use this property, but it is useful to know how to access it if needed.

```python
def training_step(self, batch, batch_idx):
    if self.global_rank == 0:
        # do something only once across all the nodes
        ...
```

### [Validation with DDP](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#validation)

It is recommended to validate on single device to ensure each sample/batch gets evaluated exactly once. 
This is helpful to make sure benchmarking for research papers is done the right way. 
Otherwise, in a multi-device setting, samples could occur duplicated when `DistributedSampler` is used, for eg. with `strategy="ddp"`.
It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.

### Logging with DDP

- https://github.com/Lightning-AI/lightning/discussions/6501

`LightningModule` -> `self.log` set `rank_zero_only == True` to make sure the value will be logged only on rank 0.
This will prevent synchronization which would produce a deadlock as not all processes would perform this log call.

What about:

- `sync_dist (bool)` – if True, reduces the metric across GPUs/TPUs. Use with care as this may lead to a significant communication overhead.
- `sync_dist_group` (Optional[Any]) – the DDP group to sync across.

### Advanced techniques (sharding, etc.)

- https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#sharded-training
- https://github.com/Lightning-AI/lightning/issues/6047
- https://github.com/Lightning-AI/lightning/discussions/8795
- https://engineering.fb.com/2021/07/15/open-source/fsdp/


### In-memory datasets

Say you want to use an in-memory dataset and you have no alternative. Then putting it in shared memory is definitely the best approach

1. Just avoid putting the logic in `setup()`. This hook runs in every process, so not desired for this use case.
2. Use a strategy like `ddp_spawn` or `ddp_fork`
3. Load and process your data in the main process (i.e., before calling `trainer.fit`).
4. Give the pointer to the in-memory dataset to your dataset/dataloader
5. After processes spawn, they inherit the memory and the training in the worker processes will reference this shared data

Docs: https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html?highlight=Sharing%20Datasets%20Across%20Process%20Boundaries#sharing-datasets-across-process-boundaries

A GNN example:

https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pytorch_lightning/gin.py

Plus:

1. Their dataset: https://github.com/pyg-team/pytorch_geometric/blob/ab78d47f134ec467129136affa7379a567ca5e9f/torch_geometric/datasets/tu_dataset.py#L12
2. A generic `InMemoryDataset`: https://github.com/pyg-team/pytorch_geometric/blob/ab78d47f134ec467129136affa7379a567ca5e9f/torch_geometric/data/in_memory_dataset.py#L13