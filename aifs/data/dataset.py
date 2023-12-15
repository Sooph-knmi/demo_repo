import os
import random
from functools import cached_property
from typing import Callable
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset

from aifs.utils.logger import get_code_logger


LOGGER = get_code_logger(__name__, debug=False)


class NativeGridDataset(IterableDataset):
    """Iterable dataset for ERA5 2D + 3D data on the native (reduced-Gaussian) grid."""

    BASE_SEED = 42

    def __init__(
        self,
        data_reader: Callable,
        rollout: int = 1,
        multistep: int = 1,
        comm_group_rank: int = 0,
        comm_group_id: int = 0,
        comm_num_groups: int = 1,
        shuffle: bool = True,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        rollout : int, optional
            length of rollout window, by default 12
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        comm_group_rank : int, optional
            process rank in the torch.distributed group (important when running on multiple GPUs), by default 0
        comm_group_id: int, optional
            device group ID, default 0
        comm_num_groups : int, optional
            total number of device groups, by default 1
        shuffle : bool, optional
            Shuffle batches, by default True

        Raises
        ------
        RuntimeError
            Multistep value cannot be negative.
        """
        self.data = data_reader

        self.rollout = rollout

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # DDP-relevant info
        self.comm_group_rank = comm_group_rank
        self.comm_num_groups = comm_num_groups
        self.comm_group_id = comm_group_id
        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: Optional[np.ndarray] = None
        self.shuffle = shuffle
        self._seed: Optional[int] = None
        self._worker_id: Optional[int] = None

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

    @cached_property
    def rng(self) -> np.random.Generator:
        """Return the random number generator."""
        assert self._seed is not None
        torch.manual_seed(self._seed)
        random.seed(self._seed)
        return np.random.default_rng(seed=self._seed)

    @cached_property
    def seed(self) -> int:
        """Return the random seed."""
        assert self._worker_id is not None
        self._seed = self.BASE_SEED * (self.comm_group_id + 1) - self._worker_id
        return self._seed

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def build_einops_dim_order(self) -> str:
        ensemble = "ensemble " if self.ensemble_size > 1 else ""
        self.ensemble_dim = 1
        return f"dates variables {ensemble}gridpoints -> dates {ensemble}gridpoints variables"

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID
        """
        self._worker_id = worker_id

        # Total number of valid ICs is dataset length minus rollout minus additional multistep inputs
        len_corrected = len(self.data) - (self.rollout + (self.multi_step - 1))

        # Divide this equally across shards (one shard per group!)
        shard_size = len_corrected // self.comm_num_groups
        shard_start = self.comm_group_id * shard_size + (self.multi_step - 1)
        shard_end = min((self.comm_group_id + 1) * shard_size, len(self.data) - self.rollout)

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)

        LOGGER.debug(
            "Worker %d (pid %d, global_rank %d, comm group %d) has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.comm_group_id,
            low,
            high,
        )

        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.debug(
            "Worker %d (pid %d, global_rank %d, comm group %d, group_rank %d) using seed %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.comm_group_id,
            self.comm_group_rank,
            self.seed,
        )

    def __iter__(self):
        """Return an iterator over the dataset.

        The datasets are retrieved by ECML Tools from zarr files. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """

        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(self.chunk_index_range, size=self.n_samples_per_worker, replace=False)
        else:
            shuffled_chunk_indices = self.chunk_index_range

        LOGGER.debug(
            "Worker pid %d, global_rank %d, comm group %d, group_rank %d using indices[0:10]: %s",
            os.getpid(),
            self.global_rank,
            self.comm_group_id,
            self.comm_group_rank,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1)
            end = i + (self.rollout + 1)

            x = self.data[start:end]
            x = rearrange(x, self.build_einops_dim_order)

            x = x[:, 0, :, :]
            sample = (torch.from_numpy(x),)
            if self.ensemble_size > 1:
                # TODO: the ensemble IC generation logic belongs in new a preprocessor class
                x_eda = x[start:i, 1:, :, :]
                sample = (torch.from_numpy(x), torch.from_numpy(x_eda))

            yield sample

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Rollout: {self.rollout}
            Multistep: {self.multi_step}
        """


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process.

    Calls WeatherBenchDataset.per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID

    Raises
    ------
    RuntimeError
        If worker_info is None
    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )
