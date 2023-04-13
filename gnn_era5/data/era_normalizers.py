from typing import Dict

import numpy as np
import torch
from torch import nn

from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__, debug=True)

# these variables have sensible ranges to we leave them untouched
DO_NOT_NORMALIZE = [
    "cos_latitude",
    "cos_longitude",
    "sin_latitude",
    "sin_longitude",
    "cos_julian_day",
    "cos_local_time",
    "sin_julian_day",
    "sin_local_time",
    "insolation",
    "lsm",
]

MAX_NORMALIZE = ["blh", "sdor", "slor", "z"]


class InputNormalizer(nn.Module):
    def __init__(self, zarr_metadata: Dict) -> None:
        super().__init__()
        self._zarr_metadata = zarr_metadata

        self._max_norm_idx = np.array(
            sorted([self._zarr_metadata["name_to_index"][var] for var in MAX_NORMALIZE]),
            dtype=np.int32,
        )

        self._max_norm = torch.from_numpy(
            np.array(
                [self._zarr_metadata["statistics_by_index"]["maximum"][vidx] for vidx in self._max_norm_idx],
                dtype=np.float32,
            )
        )[None, None, :]

        # for the rest, we map to unit gaussian or leave as-is
        self._std_norm_idx = []
        self._std_norm_mu = []
        self._std_norm_sd = []

        for vname in list(self._zarr_metadata["name_to_index"]):
            if vname not in DO_NOT_NORMALIZE and vname not in MAX_NORMALIZE:
                self._std_norm_idx.append(self._zarr_metadata["name_to_index"][vname])

        self._std_norm_idx = sorted(self._std_norm_idx)

        for vidx in self._std_norm_idx:
            self._std_norm_mu.append(self._zarr_metadata["statistics_by_index"]["mean"][vidx])
            self._std_norm_sd.append(self._zarr_metadata["statistics_by_index"]["stdev"][vidx])

        self._std_norm_idx = torch.from_numpy(np.array(self._std_norm_idx, dtype=np.int32))
        self._std_norm_mu = torch.from_numpy(np.array(self._std_norm_mu, dtype=np.float32))[None, None, :]
        self._std_norm_sd = torch.from_numpy(np.array(self._std_norm_sd, dtype=np.float32))[None, None, :]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Expects a tensor of shape [bs, latlon, nvars]; normalization done in-place
        x[..., self._max_norm_idx] = x[..., self._max_norm_idx] / self._max_norm
        x[..., self._std_norm_idx] = (x[..., self._std_norm_idx] - self._std_norm_mu) / self._std_norm_sd
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # Expects a tensor of shape [bs, latlon, nvars]; de-norm done in place
        x[..., self._max_norm_idx] = x[..., self._max_norm_idx] * self._max_norm
        x[..., self._std_norm_idx] = x[..., self._std_norm_idx] * self._std_norm_sd + self._std_norm_mu
        return x


if __name__ == "__main__":
    import zarr

    fname = "/lus/h2resw01/fws4/lb/project/ai-ml/panguweather-o96/panguweather-o96-1979-2015.zarr"
    ds_wb = zarr.open(fname, mode="r")
    normalizer = InputNormalizer(ds_wb.attrs["climetlab"])
    X = torch.rand(4, 40320, 99)

    X_ = X.clone()
    LOGGER.debug("X.sum().sqrt() = %e", X.sum().sqrt())
    X_normed = normalizer(X_)
    LOGGER.debug("X_normed.sum().sqrt() = %e", X_normed.sum().sqrt())
    assert not torch.allclose(X, X_normed)

    X_retrieved = normalizer.denormalize(X_normed)
    LOGGER.debug("X_retrieved.sum().sqrt() = %e", X_retrieved.sum().sqrt())
    assert torch.allclose(X, X_retrieved, rtol=1e-3, atol=1e-4)
