import os
import numpy as np
import zarr

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gnn_era5.data.era_dataset import ERA5NativeGridDataset, worker_init_func
from gnn_era5.data.era_batch import ERA5DataBatch, era_batch_collator
from gnn_era5.data.era_normalizers import normalize_era_data_wrapper
from gnn_era5.data.era_readers import read_era_data
from gnn_era5.utils.config import YAMLConfig
from gnn_era5.utils.logger import get_logger
from gnn_era5.utils.constants import _NORMALIZERS_2D, _DL_PREFETCH_FACTOR

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

        # load data used to transform input
        zar = zarr.open(
            os.path.join(
                self.config[f"input:training:basedir"].format(resolution=self.config["input:resolution"]),
                self.config[f"input:training:filename"].format(resolution=self.config["input:resolution"]),
            ),
            mode="r",
        )
        self._mu: np.ndarray = np.array(zar.attrs["climetlab"]["statistics_by_index"]["mean"], dtype="float32")[
            np.newaxis, :, np.newaxis
        ]
        self._sd: np.ndarray = np.array(zar.attrs["climetlab"]["statistics_by_index"]["stdev"], dtype="float32")[
            np.newaxis, :, np.newaxis
        ]
        zar = None

        self.ds_train = self._get_dataset("training")
        self.ds_valid = self._get_dataset("validation")

    def _get_dataset(self, stage: str) -> ERA5NativeGridDataset:
        return ERA5NativeGridDataset(
            fname=self._get_data_filename(stage),
            era_data_reader=read_era_data,
            era_data_normalizer=normalize_era_data_wrapper(self._mu, self._sd),
            lead_time=self.config["model:lead-time"],
            rollout=self.config["model:rollout"],
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
            # custom collator (see above)
            # collate_fn=era_batch_collator,
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

    def transfer_batch_to_device(self, batch: torch.Tensor, device: torch.device, dataloader_idx: int = 0) -> torch.Tensor:
        del dataloader_idx  # not used
        batch = batch.to(device)
        return batch


class ERA5TestDataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig) -> None:
        super().__init__()
        self.bs_test = config["model:dataloader:batch-size:test"]
        self.num_workers_test = config["model:dataloader:num-workers:test"]
        self.config = config

        # TODO: will this work correctly in multi-node runs?
        self.local_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # load data used to transform input
        zar = zarr.open(
            os.path.join(
                self.config[f"input:training:basedir"].format(resolution=self.config["input:resolution"]),
                self.config[f"input:training:filename"].format(resolution=self.config["input:resolution"]),
            ),
            mode="r",
        )
        self._mu: np.ndarray = np.array(zar.attrs["climetlab"]["statistics_by_index"]["mean"], dtype="float32")[
            np.newaxis, :, np.newaxis
        ]
        self._sd: np.ndarray = np.array(zar.attrs["climetlab"]["statistics_by_index"]["stdev"], dtype="float32")[
            np.newaxis, :, np.newaxis
        ]
        zar = None

        self.ds_test = self._get_dataset("test")
        self.ds_predict = self._get_dataset("predict")

    def _get_dataset(self, stage: str) -> ERA5NativeGridDataset:
        return ERA5NativeGridDataset(
            fname=self._get_data_filename(stage),
            era_data_reader=read_era_data,
            era_data_normalizer=normalize_era_data_wrapper(self._mu, self._sd),
            lead_time=self.config["model:lead-time"],
            rollout=self.config["model:rollout"],
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
            # number of worker processes
            num_workers=num_workers,
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=True,
            # custom collator (see above)
            # collate_fn=era_batch_collator,
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches (default prefetch_factor == 2)
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

    def transfer_batch_to_device(self, batch: torch.Tensor, device: torch.device, dataloader_idx: int = 0) -> torch.Tensor:
        del dataloader_idx  # not used
        batch = batch.to(device)
        return batch
