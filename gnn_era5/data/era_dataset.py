import os
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset, get_worker_info
from zarr.core import Array

from gnn_era5.utils.constants import _DL_PREFETCH_FACTOR, _ERA_PLEV
from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__, debug=False)


class ERA5NativeGridDataset(IterableDataset):
    """
    Iterable dataset for ERA5 2D + 3D data on the native (reduced-Gaussian) grid.
    """

    def __init__(
        self,
        fname_2d: str,
        fname_3d: str,
        era_2d_data_reader: Callable,
        era_3d_data_reader: Callable,
        era_2d_data_normalizer: Callable,
        era_3d_data_normalizer: Callable,
        lead_time: int = 6,
        rollout: int = 4,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initialize (part of) the dataset state.
        Args:
            fname_2d / 3d: zarr file name with 2D / 3D data
            era_[2d|3d]_data_reader: user function that opens and returns the zarr array data
            lead_time: lead time (multiple of 6 hours!)
            rollout: length of rollout window (Keisler, 2021)
            rank: process rank in the torch.distributed context (important when running on multiple GPUs)
            world_size: total number of processes (nodes * GPUs_per_node) in the torch.distributed context
        """
        self.fname_2d = fname_2d
        self.fname_3d = fname_3d

        self.ds_3d: Optional[Array] = None
        self.ds_2d: Optional[Array] = None

        self.lead_time = lead_time
        assert self.lead_time > 0 and self.lead_time % 6 == 0, "Lead time must be multiple of 6 hours"
        self.lead_step = lead_time // 6

        LOGGER.debug("Dataset lead_time = %d, lead_step = %d ...", self.lead_time, self.lead_step)

        self.rollout = rollout

        self.nlev = _ERA_PLEV

        self._read_2d_era = era_2d_data_reader
        self._read_3d_era = era_3d_data_reader

        self._normalize_2d_era = era_2d_data_normalizer
        self._normalize_3d_era = era_3d_data_normalizer

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

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of WeatherBenchDataset after the worker process has been spawned."""
        if self.ds_2d is None:
            self.ds_2d = self._read_2d_era(self.fname_2d)
        if self.ds_3d is None:
            self.ds_3d = self._read_3d_era(self.fname_3d)

        # sanity check
        assert self.ds_2d.shape[0] == self.ds_3d.shape[0], "The 2d and 3d ERA datasets do not have the same no of time points!"

        shard_size = int(np.floor(self.ds_2d.shape[0] / self.world_size))
        shard_start, shard_end = self.rank * shard_size, min((self.rank + 1) * shard_size, self.ds_2d.shape[0])

        ds_len = shard_end - shard_start - self.rollout
        self.n_samples_per_worker = ds_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)

        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.debug(
            "Worker PID %d has access to shard (%i to %i), with ds_len = %i, n_chunks_per_worker = %i ",
            os.getpid(),
            shard_start,
            shard_end,
            ds_len,
            self.n_samples_per_worker,
        )

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        self.rng = np.random.default_rng(seed=torch.initial_seed())

    def __iter__(self):
        # this needs to happen at the start of every epoch
        shuffled_chunk_indices = self.rng.choice(self.chunk_index_range, size=self.n_samples_per_worker, replace=False)

        for i in shuffled_chunk_indices:
            start, end = i, i + (self.rollout + 1) * self.lead_step
            LOGGER.debug(
                "Worker PID %d selected start-end range [%i, %i] with stride lead_step = %i",
                os.getpid(),
                start,
                end,
                self.lead_step,
            )

            X2d = self._normalize_2d_era(self.ds_2d[start : end : self.lead_step])
            X2d = rearrange(X2d, "r var latlon -> r latlon var")
            X3d = self._normalize_3d_era(self.ds_3d[start : end : self.lead_step])
            X3d = rearrange(X3d, "r var lev latlon -> r latlon (var lev)")
            LOGGER.debug("Worker PID %d has arrays X2d, X3d with shapes = %s, %s", os.getpid(), X2d.shape, X3d.shape)

            X = np.concatenate([X3d, X2d], axis=-1)
            LOGGER.debug("Worker PID %d produced a sample of size %s", os.getpid(), X.shape)

            yield (torch.from_numpy(X), start)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Filename-2D: {str(self.fname_2d)}
            Filename-3D: {str(self.fname_3d)}
            Lead time: {self.lead_time}
            Rollout: {self.rollout}
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

    from gnn_era5.data.era_datamodule import era_batch_collator, read_2d_era_data, read_3d_era_data
    from gnn_era5.utils.config import YAMLConfig

    _ROLLOUT = 2
    config = YAMLConfig("/home/syma/dask/codes/gnn-era5/gnn_era5/config/atos.yaml")

    def get_data_filename(type_: str, cfg_: YAMLConfig) -> str:
        # type == [pl | sfc]
        return os.path.join(
            cfg_[f"input:{type_}:validation:basedir"].format(resolution=cfg_["input:resolution"]),
            cfg_[f"input:{type_}:validation:filename"].format(resolution=cfg_["input:resolution"]),
        )

    # dummy normalizers
    def normalize_2d_era_data(data: np.ndarray) -> np.ndarray:
        return data

    def normalize_3d_era_data(data: np.ndarray) -> np.ndarray:
        return data

    era5_ds = ERA5NativeGridDataset(
        fname_2d=get_data_filename("sfc", config),
        fname_3d=get_data_filename("pl", config),
        era_2d_data_reader=read_2d_era_data,
        era_3d_data_reader=read_3d_era_data,
        era_2d_data_normalizer=normalize_2d_era_data,
        era_3d_data_normalizer=normalize_3d_era_data,
        lead_time=config["model:lead-time"],
        rollout=_ROLLOUT,
        rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=config["model:num-gpus"] * config["model:num-nodes"],
    )

    era5_dl = DataLoader(
        era5_ds,
        batch_size=8,
        num_workers=8,
        collate_fn=era_batch_collator,
        # worker initializer
        worker_init_fn=worker_init_func,
        # prefetch batches (default prefetch_factor == 2)
        prefetch_factor=_DL_PREFETCH_FACTOR,
        persistent_workers=True,
    )

    # simple dataloader speed test
    for idx_batch, batch in enumerate(era5_dl):
        LOGGER.info("Batch index: %d - X.shape: %s idx: %s", idx_batch, batch.X.shape, batch.idx)
        assert len(batch) == (_ROLLOUT + 1)
        for r in range(len(batch)):
            LOGGER.debug("Rollout step %d: batch.X.shape = %s", r, batch.X[:, r, ...].shape)
        if idx_batch > 16:
            break
