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
        model_comm_group_rank: int = 0,
        model_comm_group_id: int = 0,
        model_comm_num_groups: int = 1,
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
        model_comm_group_rank : int, optional
            process rank in the torch.distributed group (important when running on multiple GPUs), by default 0
        model_comm_group_id: int, optional
            device group ID, default 0
        model_comm_num_groups : int, optional
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
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_id = model_comm_group_id
        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: Optional[np.ndarray] = None
        self.shuffle = shuffle

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

    @cached_property
    def rng(self) -> np.random.Generator:
        """Return the random number generator."""
        return np.random.default_rng(seed=self.seed)

    @cached_property
    def seed(self) -> int:
        """Return the random seed."""
        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        if "PL_SEED_WORKERS" in os.environ:
            self.seed_worker_random_gen(self.worker_id, self.rank)
            return np.random.randint(low=0, high=2**30)

        return torch.initial_seed()

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
        self.worker_id = worker_id

        # Total number of valid ICs is dataset length minus rollout minus additional multistep inputs
        len_corrected = len(self.data) - (self.rollout + (self.multi_step - 1))

        # Divide this equally across shards (one shard per group!)
        shard_size = len_corrected // self.model_comm_num_groups
        shard_start = self.model_comm_group_id * shard_size + (self.multi_step - 1)
        shard_end = min((self.model_comm_group_id + 1) * shard_size, len(self.data) - self.rollout)

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

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        # if "PL_SEED_WORKERS" in os.environ:
        #     self.seed_worker_random_gen(worker_id, self.model_comm_group_rank)
        #     seed = np.random.randint(low=0, high=2**30)
        # else:
        #     seed = torch.initial_seed()
        # seed = self.BASE_SEED * self.model_comm_group_id + self.model_comm_group_rank

        # self.seed_worker_random_gen(worker_id, self.model_comm_group_rank)
        seed = self.BASE_SEED * (self.comm_group_id + 1) - worker_id  # np.random.randint(low=0, high=2**30)
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)

        LOGGER.debug(
            "Worker %d (pid %d, global_rank %d, comm group %d, group_rank %d) using seed %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.comm_group_id,
            self.comm_group_rank,
            seed,
        )

    def seed_worker_random_gen(self, worker_id: int, rank: Optional[int] = None) -> None:
        # taken from https://github.com/Lightning-AI/lightning/blob/master/src/lightning/fabric/utilities/seed.py
        # https://github.com/Lightning-AI/lightning/blob/master/LICENSE
        """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
        ``seed_everything(seed, workers=True)``.

        See also the PyTorch documentation on
        `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
        """
        # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        model_comm_group_rank = rank if rank is not None else 0
        # process_seed = torch.initial_seed()
        # back out the base seed so we can use all the bits
        base_seed = self.BASE_SEED * (self.comm_group_id + 1) - worker_id
        LOGGER.debug(
            "Initializing random number generators of group_rank %d worker %d group %d global_rank %d with base seed %d",
            model_comm_group_rank,
            worker_id,
            self.comm_group_id,
            self.global_rank,
            base_seed,
        )
        ss = np.random.SeedSequence([base_seed, worker_id, model_comm_group_rank])
        # use 128 bits (4 x 32-bit words)
        np.random.seed(ss.generate_state(4))
        # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
        torch_ss, stdlib_ss = ss.spawn(2)
        torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
        # use 128 bits expressed as an integer
        stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
        random.seed(stdlib_seed)

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
                x_eda = x[:, 1:, :, :]
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
