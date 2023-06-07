import os
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset, get_worker_info
from zarr.core import Array

from aifs.utils.constants import _DL_PREFETCH_FACTOR, _ERA_PLEV
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__, debug=False)


class ERA5NativeGridDataset(IterableDataset):
    """
    Iterable dataset for ERA5 2D + 3D data on the native (reduced-Gaussian) grid.
    """

    def __init__(
        self,
        fname: str,
        era_data_reader: Callable,
        lead_time: int = 6,
        rollout: int = 4,
        multistep: int = 1,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Initialize (part of) the dataset state.
        Args:
            fname / 3d: zarr file name with 2D / 3D data
            era_[2d|3d]_data_reader: user function that opens and returns the zarr array data
            lead_time: lead time (multiple of 6 hours!)
            rollout: length of rollout window (Keisler, 2021)
            multistep: collate (t-1, ... t - multistep) into the input state vector,
            rank: process rank in the torch.distributed context (important when running on multiple GPUs)
            world_size: total number of processes (nodes * GPUs_per_node) in the torch.distributed context
        """
        self.fname = fname
        self.ds: Optional[Array] = None

        self.lead_time = lead_time
        assert self.lead_time > 0 and self.lead_time % 6 == 0, "Lead time must be multiple of 6 hours"
        self.lead_step = lead_time // 6

        LOGGER.debug("Dataset lead_time = %d, lead_step = %d ...", self.lead_time, self.lead_step)

        self.rollout = rollout

        self.nlev = _ERA_PLEV

        self._read_era = era_data_reader

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0
        self.rng: Optional[int] = None

        # DDP-relevant info
        self.rank = rank
        self.world_size = world_size

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: Optional[np.ndarray] = None
        self.shuffle = shuffle

        self.mstep = multistep
        if self.mstep <= 0:
            LOGGER.error("Multistep value invalid %d - check your configuration file!", self.mstep)
            raise RuntimeError

        self.debug = debug

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of WeatherBenchDataset after the worker process has been spawned."""
        if self.ds is None:
            self.ds = self._read_era(self.fname)

        shard_size = int(np.floor(self.ds.shape[0] / self.world_size))
        shard_start, shard_end = self.rank * shard_size, min((self.rank + 1) * shard_size, self.ds.shape[0])

        if self.debug:
            LOGGER.info(
                "Worker PID %d: Device %d operates on a shard range [%i, %i] of length = %i",
                os.getpid(),
                self.rank,
                shard_start,
                shard_end,
                shard_end - shard_start,
            )

        if self.rank == 0:
            # shift start position to have sufficient samples for multistep input
            shard_start = (self.mstep - 1) * self.lead_step
        ds_len = shard_end - shard_start - self.rollout
        self.n_samples_per_worker = ds_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)

        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.debug(
            "Worker PID %d has access to shard (%i to %i), with ds_len = %i, n_chunks_per_worker = %i, multistep = %d",
            os.getpid(),
            shard_start,
            shard_end,
            ds_len,
            self.n_samples_per_worker,
            self.mstep,
        )

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        self.rng = np.random.default_rng(seed=torch.initial_seed())

    def __iter__(self):
        # this needs to happen at the start of every epoch
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(self.chunk_index_range, size=self.n_samples_per_worker, replace=False)
        else:
            shuffled_chunk_indices = self.chunk_index_range

        for i in shuffled_chunk_indices:
            start, end = i - (self.mstep - 1) * self.lead_step, i + (self.rollout + 1) * self.lead_step
            LOGGER.debug(
                "Worker PID %d selected start-end range [%i, %i] with stride lead_step = %i",
                os.getpid(),
                start,
                end,
                self.lead_step,
            )
            if self.debug:
                LOGGER.info(
                    "Worker PID %d serving device %d selected start-end range [%i, %i] with stride lead_step = %i",
                    os.getpid(),
                    self.rank,
                    start,
                    end,
                    self.lead_step,
                )

            X = self.ds[start : end : self.lead_step]
            X = rearrange(X, "r var latlon -> r latlon var")
            LOGGER.debug("Worker PID %d produced a sample of size %s", os.getpid(), X.shape)
            if self.debug:
                LOGGER.info("Worker PID %d serving device %d produced a sample of size %s", os.getpid(), self.rank, X.shape)
            yield torch.from_numpy(X)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Filename: {str(self.fname)}
            Lead time: {self.lead_time}
            Rollout: {self.rollout}
            Multistep: {self.mstep}
        """


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process by calling WeatherBenchDataset.per_worker_init()."""
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from aifs.data.era_readers import read_era_data
    from aifs.utils.config import YAMLConfig

    _ROLLOUT = 2
    _MULTISTEP = 2
    config = YAMLConfig("aifs/config/atos.yaml")

    def _get_data_filename(stage: str) -> str:
        # field_type == [pl | sfc], stage == [training | validation]
        return os.path.join(
            config[f"input:{stage}:basedir"].format(resolution=config["input:resolution"]),
            config[f"input:{stage}:filename"].format(resolution=config["input:resolution"]),
        )

    era5_ds = ERA5NativeGridDataset(
        fname=_get_data_filename("validation"),
        era_data_reader=read_era_data,
        lead_time=config["model:lead-time"],
        rollout=_ROLLOUT,
        multistep=_MULTISTEP,
        rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=config["model:num-gpus"] * config["model:num-nodes"],
    )

    era5_dl = DataLoader(
        era5_ds,
        batch_size=8,
        num_workers=8,
        worker_init_fn=worker_init_func,
        prefetch_factor=_DL_PREFETCH_FACTOR,
        persistent_workers=True,
    )

    # simple dataloader speed test
    for idx_batch, batch in enumerate(era5_dl):
        LOGGER.info("Batch index: %d - shape: %s", idx_batch, batch.shape)
        assert batch.shape[1] == (_ROLLOUT + _MULTISTEP)
        for r in range(batch.shape[1]):
            LOGGER.debug("Rollout step %d: batch.shape = %s", r, batch[:, r, ...].shape)
        if idx_batch > 4:
            break
