from typing import List, Tuple

from einops import rearrange
import numpy as np
import torch

from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__)


class ERA5DataBatch:
    """
    Custom batch type for ERA5 data on the native (reduced Gaussian) grid.
    Contains two elements:
        X: tensor storing the batch data, shape (batch_size, rollout + 1, latlon, nvar)
           len(X) == rollout + 1
        idx: global sample indices, so we can map any sample back to its correct index in the original dataset
    """

    def __init__(self, batch_data) -> None:
        zipped_batch = list(zip(*batch_data))
        self.X = torch.stack(zipped_batch[0], dim=0)
        self.idx = torch.from_numpy(np.array(zipped_batch[1], dtype=np.int32))

    def pin_memory(self):
        """Custom memory pinning. See https://pytorch.org/docs/stable/data.html#memory-pinning"""
        self.X = self.X.pin_memory()
        self.idx = self.idx.pin_memory()
        return self

    def __len__(self) -> int:
        # returns the length of the rollout window
        # TODO: is this really needed?
        return self.X.shape[1]


def era_batch_collator(batch_data):
    """
    Custom collation function. It collates several batch chunks into a "full" batch.
    """
    return ERA5DataBatch(batch_data=batch_data)
