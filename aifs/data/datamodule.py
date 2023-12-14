import os
from functools import cached_property
from typing import Dict

import pytorch_lightning as pl
from ecml_tools.data import open_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from aifs.data.dataindices import IndexCollection
from aifs.data.dataset import NativeGridDataset
from aifs.data.dataset import worker_init_func
from aifs.utils.logger import get_code_logger


LOGGER = get_code_logger(__name__)


class ECMLDataModule(pl.LightningDataModule):
    """ECML data module for PyTorch Lightning."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize ECML data module.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        """
        super().__init__()
        self.config = config

        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))  # global rank
        self.model_comm_group_id = (
            self.global_rank // self.config.hardware.num_gpus_per_model
        )  # id of the model communication group the rank is participating in
        self.model_comm_group_rank = (
            self.global_rank % self.config.hardware.num_gpus_per_model
        )  # rank within one model communication group
        total_gpus = self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes
        assert (
            total_gpus
        ) % self.config.hardware.num_gpus_per_model == 0, (
            f"GPUs per model {self.config.hardware.num_gpus_per_model} does not divide total GPUs {total_gpus}"
        )
        self.model_comm_num_groups = (
            self.config.hardware.num_gpus_per_node * self.config.hardware.num_nodes // self.config.hardware.num_gpus_per_model
        )  # number of model communication groups
        LOGGER.debug(
            "Rank %d model communication group number %d, with local model communication group rank %d",
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
        )

        # Set the maximum rollout to be expected
        self.rollout = (
            self.config.training.rollout.max
            if self.config.training.rollout.epoch_increment > 0
            else self.config.training.rollout.start
        )

    def _check_resolution(self, resolution) -> None:
        assert (
            self.config.data.resolution.lower() == resolution.lower()
        ), f"Network resolution {self.config.data.resolution=} does not match dataset resolution {resolution=}"

    @cached_property
    def statistics(self) -> Dict:
        return self.ds_train.statistics

    @cached_property
    def metadata(self) -> Dict:
        return self.ds_train.metadata

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.ds_train.name_to_index)

    @cached_property
    def ds_train(self) -> NativeGridDataset:
        return self._get_dataset(open_dataset(OmegaConf.to_container(self.config.dataloader.training, resolve=True)))

    @cached_property
    def ds_valid(self) -> NativeGridDataset:
        r = self.rollout
        if self.config.diagnostics.eval.enabled:
            r = max(r, self.config.diagnostics.eval.rollout)
        assert self.config.dataloader.training.end < self.config.dataloader.validation.start, (
            f"Training end date {self.config.dataloader.training.end} is not before"
            f"validation start date {self.config.dataloader.validation.start}"
        )
        return self._get_dataset(
            open_dataset(OmegaConf.to_container(self.config.dataloader.validation, resolve=True)),
            shuffle=False,
            rollout=r,
        )

    @cached_property
    def ds_test(self) -> NativeGridDataset:
        assert self.config.dataloader.training.end < self.config.dataloader.test.start, (
            f"Training end date {self.config.dataloader.training.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        assert self.config.dataloader.validation.end < self.config.dataloader.test.start, (
            f"Validation end date {self.config.dataloader.validation.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        return self._get_dataset(open_dataset(OmegaConf.to_container(self.config.dataloader.test, resolve=True)), shuffle=False)

    def _get_dataset(self, data_reader, shuffle: bool = True, rollout: int = 1) -> NativeGridDataset:
        r = max(rollout, self.rollout)
        data = NativeGridDataset(
            data_reader=data_reader,
            rollout=r,
            multistep=self.config.training.multistep_input,
            model_comm_group_rank=self.model_comm_group_rank,
            model_comm_group_id=self.model_comm_group_id,
            model_comm_num_groups=self.model_comm_num_groups,
            shuffle=shuffle,
        )
        self._check_resolution(data.resolution)
        return data

    def _get_dataloader(self, ds: NativeGridDataset, stage: str) -> DataLoader:
        assert stage in ["training", "validation", "test"]
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            # number of worker processes
            num_workers=self.config.dataloader.num_workers[stage],
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
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")
