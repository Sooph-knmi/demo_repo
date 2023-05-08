import os

import pytorch_lightning as pl
import zarr
from torch.utils.data import DataLoader

from aifs.data.era_dataset import ERA5NativeGridDataset, worker_init_func
from aifs.data.era_readers import read_era_data
from aifs.utils.config import YAMLConfig
from aifs.utils.constants import _DL_PREFETCH_FACTOR
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


class ERA5DataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig) -> None:
        super().__init__()
        self.bs_train = config["model:dataloader:batch-size:training"]
        self.bs_val = config["model:dataloader:batch-size:validation"]

        self.num_workers_train = config["model:dataloader:num-workers:training"]
        self.num_workers_val = config["model:dataloader:num-workers:validation"]
        self.config = config

        # TODO: will this work correctly in multi-node runs?
        self.local_rank = int(os.environ.get("SLURM_PROCID", "0"))

        self.ds_train = self._get_dataset("training")
        self.ds_valid = self._get_dataset("validation")

        ds_tmp = zarr.open(self._get_data_filename("training"), mode="r")
        self.input_metadata = ds_tmp.attrs["climetlab"]
        ds_tmp = None

    def _get_dataset(self, stage: str) -> ERA5NativeGridDataset:
        return ERA5NativeGridDataset(
            fname=self._get_data_filename(stage),
            era_data_reader=read_era_data,
            lead_time=self.config["model:lead-time"],
            rollout=self.config["model:rollout"],
            multistep=self.config["model:multistep-input"],
            rank=self.local_rank,
            world_size=self.config["model:num-gpus"] * self.config["model:num-nodes"],
        )

    def _get_data_filename(self, stage: str) -> str:
        # field_type == [pl | sfc], stage == [training | validation]
        return os.path.join(
            self.config[f"input:{stage}:basedir"].format(resolution=self.config["input:resolution"]),
            self.config[f"input:{stage}:filename"].format(resolution=self.config["input:resolution"]),
        )

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
            # prefetch batches (default prefetch_factor == 2)
            prefetch_factor=_DL_PREFETCH_FACTOR,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, self.num_workers_train, self.bs_train)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, self.num_workers_val, self.bs_val)


class ERA5TestDataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig) -> None:
        super().__init__()
        self.bs_test = config["model:dataloader:batch-size:test"]
        self.num_workers_test = config["model:dataloader:num-workers:test"]
        self.config = config

        # TODO: will this work correctly in multi-node runs?
        self.local_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # load data used to transform input
        ds_tmp = zarr.open(self._get_data_filename("training"), mode="r")
        self.input_metadata = ds_tmp.attrs["climetlab"]
        ds_tmp = None

        self.ds_test = self._get_dataset("test")
        self.ds_predict = self._get_dataset("predict")

    def _get_dataset(self, stage: str) -> ERA5NativeGridDataset:
        return ERA5NativeGridDataset(
            fname=self._get_data_filename(stage),
            era_data_reader=read_era_data,
            lead_time=self.config["model:lead-time"],
            rollout=self.config["model:rollout"],
            multistep=self.config["model:multistep-input"],
            rank=self.local_rank,
            world_size=self.config["model:num-gpus"] * self.config["model:num-nodes"],
            shuffle=False,
        )

    def _get_data_filename(self, stage: str) -> str:
        # field_type == [pl | sfc], stage == [training | validation | test]
        return os.path.join(
            self.config[f"input:{stage}:basedir"].format(resolution=self.config["input:resolution"]),
            self.config[f"input:{stage}:filename"].format(resolution=self.config["input:resolution"]),
        )

    def _get_dataloader(self, ds: ERA5NativeGridDataset, num_workers: int, batch_size: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_func,
            prefetch_factor=_DL_PREFETCH_FACTOR,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError("The ERA5TestDataModule should be used for inference (test-mode) only!")

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError("The ERA5TestDataModule should be used for inference (test-mode) only!")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, self.num_workers_test, self.bs_test)

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_predict, self.num_workers_test, self.bs_test)
