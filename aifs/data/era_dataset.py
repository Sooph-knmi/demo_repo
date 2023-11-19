import os
import random
import re
from typing import Callable
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from zarr.core import Array

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)


class ERA5NativeGridDataset(IterableDataset):
    """Iterable dataset for ERA5 2D + 3D data on the native (reduced-Gaussian) grid."""

    BASE_SEED = 42

    def __init__(
        self,
        fname_an: str,
        data_reader: Callable,
        fname_eda: Optional[str] = None,
        lead_time: int = 6,
        rollout: int = 4,
        multistep: int = 1,
        comm_group_rank: int = 0,
        comm_group_id: int = 0,
        comm_num_groups: int = 1,
        shuffle: bool = True,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        fname_an, fname_eda : str, the latter optional
            zarr file names (analysis and EDA) with 2D / 3D data
            if fname_eda is None, then no EDA data is read or used
        data_reader : Callable
            user function that opens and returns the zarr array data
        lead_time : int, optional
            lead time (multiple of 6 hours!), by default 6
        rollout : int, optional
            length of rollout window (Keisler, 2021), by default 4
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
        self.fname_an = fname_an
        self.fname_eda = fname_eda

        # master switch for EDA
        self.eda = self.fname_eda is not None

        LOGGER.info("Are we using the EDA dataset? %s", self.eda)
        LOGGER.info("Analysis zarr filename: %s", self.fname_an)
        LOGGER.info("EDA zarr filename: %s", self.fname_eda if self.eda else "N/A")

        self.ds_an: Optional[Array] = None
        self.ds_eda: Optional[Array] = None

        self.lead_time = lead_time
        # Data_step should be stored in meta-data of file
        self.data_step = int(re.findall(r"\d+", self.fname_an)[-1])
        assert self.data_step == 6 or self.data_step == 1, f"Data step detected as {self.data_step}, only 1 and 6 are supported"
        assert self.lead_time > 0 and self.lead_time % self.data_step == 0, f"Lead time must be multiple of {self.data_step} hours"
        self.lead_step = lead_time // self.data_step

        LOGGER.debug("Dataset lead_time = %d, lead_step = %d ..., date_step = %d", self.lead_time, self.lead_step, self.data_step)

        self.rollout = rollout

        self._read_era = data_reader

        # lazy init
        self.rng: Optional[int] = None

        # DDP-relevant info
        self.comm_group_rank = comm_group_rank
        self.comm_num_groups = comm_num_groups
        self.comm_group_id = comm_group_id
        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: Optional[np.ndarray] = None
        self.shuffle = shuffle

        self.multi_step = multistep
        if self.multi_step <= 0:
            LOGGER.error(
                "Multistep value invalid %d - check your configuration file!",
                self.multi_step,
            )
            raise RuntimeError

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of WeatherBenchDataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID
        """

        if self.ds_an is None:
            self.ds_an = self._read_era(self.fname_an)
        if self.ds_eda is None and self.eda:
            self.ds_eda = self._read_era(self.fname_eda).data  # the EDA dataset stores its data under .data
            LOGGER.debug("ds_an.shape = %s, ds_eda.shape = %s", self.ds_an.shape, self.ds_eda.shape)
            assert (
                self.ds_an.shape[0] == self.ds_eda.shape[0]
            ), f"The ERA5 analysis and EDA datasets have different lengths! {self.ds_an.shape[0]} != {self.ds_eda.shape[0]}"

        # Total number of valid ICs is dataset length
        # minus rollout
        # minus additional multistep inputs
        ds_total_len = self.ds_an.shape[0] - (self.rollout + (self.multi_step - 1)) * self.lead_step
        # Divide this equally across shards (one shard per group!)
        shard_size = int(np.floor(ds_total_len / self.comm_num_groups))
        shard_start = self.comm_group_id * shard_size + (self.multi_step - 1) * self.lead_step
        shard_end = min((self.comm_group_id + 1) * shard_size, self.ds_an.shape[0] - self.rollout * self.lead_step)

        ds_len = shard_end - shard_start
        self.n_samples_per_worker = ds_len // n_workers

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
        # this needs to happen at the start of every epoch
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
            start, end = (
                i - (self.multi_step - 1) * self.lead_step,
                i + (self.rollout + 1) * self.lead_step,
            )
            X = self.ds_an[start : end : self.lead_step]
            X = rearrange(X, "r var latlon -> r latlon var")
            if self.eda:
                start_ens, end_ens = (
                    i - (self.multi_step - 1) * self.lead_step,
                    i + self.lead_step,
                )
                X_ens = self.ds_eda[start_ens : end_ens : self.lead_step]
                X_ens = rearrange(X_ens, "s var e latlon -> s latlon var e")
                sample = (torch.from_numpy(X), torch.from_numpy(X_ens))
            else:
                sample = (torch.from_numpy(X),)
            yield sample

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Analysis filename: {str(self.fname_an)}
            Lead time: {self.lead_time}
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
