from typing import Dict

import numpy as np
import torch
from torch import nn

from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__, debug=True)

# these variables have sensible ranges so we leave them untouched
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

MAX_NORMALIZE = ["sdor", "slor", "z"]


class InputNormalizer(nn.Module):
    def __init__(self, zarr_metadata: Dict) -> None:
        super().__init__()
        self._zarr_metadata = zarr_metadata

        _max_norm_idx = np.array(
            sorted([self._zarr_metadata["name_to_index"][var] for var in MAX_NORMALIZE]),
            dtype=np.int32,
        )

        _max_norm = torch.from_numpy(
            np.array(
                [self._zarr_metadata["statistics_by_index"]["maximum"][vidx] for vidx in _max_norm_idx],
                dtype=np.float32,
            )
        )

        # for the rest, we map to unit gaussian or leave as-is
        _std_norm_idx = []
        _std_norm_mu = []
        _std_norm_sd = []

        for vname in list(self._zarr_metadata["name_to_index"]):
            if vname not in DO_NOT_NORMALIZE and vname not in MAX_NORMALIZE:
                _std_norm_idx.append(self._zarr_metadata["name_to_index"][vname])

        _std_norm_idx = sorted(_std_norm_idx)

        for vidx in _std_norm_idx:
            _std_norm_mu.append(self._zarr_metadata["statistics_by_index"]["mean"][vidx])
            _std_norm_sd.append(self._zarr_metadata["statistics_by_index"]["stdev"][vidx])

        # register all buffers - this will ensure they get copied to the correct device(s)
        self.register_buffer("_max_norm_idx", torch.from_numpy(_max_norm_idx), persistent=True)
        self.register_buffer("_max_norm", _max_norm, persistent=True)
        self.register_buffer("_std_norm_idx", torch.from_numpy(np.array(_std_norm_idx, dtype=np.int32)), persistent=True)
        self.register_buffer("_std_norm_mu", torch.from_numpy(np.array(_std_norm_mu, dtype=np.float32)), persistent=True)
        self.register_buffer("_std_norm_sd", torch.from_numpy(np.array(_std_norm_sd, dtype=np.float32)), persistent=True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes an input tensor x of shape [..., nvars]; normalization done in-place."""
        x[..., self._max_norm_idx] = x[..., self._max_norm_idx] / self._max_norm
        x[..., self._std_norm_idx] = (x[..., self._std_norm_idx] - self._std_norm_mu) / self._std_norm_sd
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalizes an input tensor x of shape [..., nvars | nvars_pred]; normalization done in-place."""
        # input and predicted tensors have different shapes
        # hence, we mask out the indices >= x.shape[-1] - i.e. the variables that are not predicted
        max_denorm_mask = self._max_norm_idx < x.shape[-1]
        std_denorm_mask = self._std_norm_idx < x.shape[-1]

        x[..., self._max_norm_idx[max_denorm_mask]] = x[..., self._max_norm_idx[max_denorm_mask]] * self._max_norm[max_denorm_mask]
        x[..., self._std_norm_idx[std_denorm_mask]] = (
            x[..., self._std_norm_idx[std_denorm_mask]] * self._std_norm_sd[std_denorm_mask] + self._std_norm_mu[std_denorm_mask]
        )
        return x


if __name__ == "__main__":
    import zarr

    fname = "/lus/h2resw01/fws4/lb/project/ai-ml/panguweather-o96/panguweather-o96-1979-2015.zarr"
    ds_wb = zarr.open(fname, mode="r")
    normalizer = InputNormalizer(ds_wb.attrs["climetlab"])
    X = torch.rand(4, 40320, 99, dtype=torch.float32) * 2.0

    X_ = X.clone()
    LOGGER.debug("X.sum().sqrt() = %e", X.sum().sqrt())
    X_norm = normalizer(X_)
    LOGGER.debug("X_norm.sum().sqrt() = %e", X_norm.sum().sqrt())
    assert not torch.allclose(X, X_norm)

    X_denorm = normalizer.denormalize(X_norm.clone())
    LOGGER.debug("X_denorm.sum().sqrt() = %e", X_denorm.sum().sqrt())
    # we don't get back the original X _exactly_, but should be close enough
    LOGGER.debug("max |X - X_denorm| = %e", torch.max(torch.abs(X - X_denorm)))
    assert torch.allclose(X, X_denorm, rtol=1e-3, atol=5e-2)
