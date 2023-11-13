import math
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import zarr
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from aifs.data.era_dataset import ERA5NativeGridDataset
from aifs.data.era_dataset import worker_init_func
from aifs.data.era_readers import read_era_data
from aifs.utils.logger import get_code_logger


LOGGER = get_code_logger(__name__)


class ERA5DataModule(pl.LightningDataModule):
    """ERA5 data module for PyTorch Lightning."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize ERA5 data module.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        """
        super().__init__()
        self.bs_train = config.dataloader.batch_size.training
        self.bs_val = config.dataloader.batch_size.validation

        self.num_workers_train = config.dataloader.num_workers.training
        self.num_workers_val = config.dataloader.num_workers.validation
        self.config = config

        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))  # global rank
        self.model_comm_group_id = (
            self.global_rank // self.config.hardware.num_gpus_per_model
        )  # id of the model communication group the rank is participating in
        self.model_comm_group_rank = (
            self.global_rank % self.config.hardware.num_gpus_per_model
        )  # rank within one model communication group
        self.model_comm_num_groups = math.ceil(
            self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes / self.config.hardware.num_gpus_per_model
        )  # number of model communication groups
        LOGGER.debug(
            "Rank %d model communication group number %d, with local model communication group rank %d",
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
        )

        self.ds_train = self._get_dataset("training", shuffle=True)

        r = self.config.training.rollout.max
        if config.diagnostics.eval.enabled:
            r = max(r, config.diagnostics.eval.rollout)
        self.ds_valid = self._get_dataset("validation", shuffle=False, rollout=r)

        ds_tmp = zarr.open(self._get_data_file_path("an", "training"), mode="r")
        self.input_metadata = ds_tmp.attrs["climetlab"]
        ds_tmp = None

    def _get_dataset(self, stage: str, shuffle: bool = True, rollout: Optional[int] = None) -> ERA5NativeGridDataset:
        rollout_config = (
            self.config.training.rollout.max
            if self.config.training.rollout.epoch_increment > 0
            else self.config.training.rollout.start
        )
        r = max(rollout, rollout_config) if rollout is not None else rollout_config
        return ERA5NativeGridDataset(
            fname_an=self._get_data_file_path("an", stage),
            data_reader=read_era_data,
            fname_eda=self._get_data_file_path("eda", stage) if self.config.training.eda_initial_perturbations else None,
            lead_time=self.config.training.lead_time,
            rollout=r,
            multistep=self.config.training.multistep_input,
            model_comm_group_rank=self.model_comm_group_rank,
            model_comm_group_id=self.model_comm_group_id,
            model_comm_num_groups=self.model_comm_num_groups,
            shuffle=shuffle,
        )

    def _get_data_file_path(self, type_: str, stage: str) -> str:
        # type_ == [an | eda], stage == [training | validation]
        fpath = Path(
            self.config.hardware.paths[stage][type_],
            self.config.hardware.files[stage][type_],
        )
        return fpath.as_posix()

    def _get_dataloader(self, ds: ERA5NativeGridDataset, num_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            # number of worker processes
            num_workers=num_workers,
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=True,
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, self.num_workers_train, self.bs_train)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, self.num_workers_val, self.bs_val)
