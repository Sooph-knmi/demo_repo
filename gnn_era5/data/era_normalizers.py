import os
from typing import Callable, List

import numpy as np

from gnn_era5.utils.logger import get_logger

LOGGER = get_logger(__name__, debug=False)


# ---------------------------------------------------------------------------------
# Normalizer logic
# - surface (2D) vars: min-max or (mu, sd) normalization
# - plev (3D) vars: (mu, sd) normalization only
# All statistics need to be precomputed and available to pass to the wrappers
# ---------------------------------------------------------------------------------
def normalize_2d_era_data_wrapper(norm_methods: List[str], stats_2d: np.ndarray) -> Callable:
    def _normalize_2d_era_data(data: np.ndarray) -> np.ndarray:
        LOGGER.debug("Worker %d produced 2D var data.shape: %s", os.getpid(), data.shape)
        # 2D data.shape = (rollout, nvar, latlon)
        for vidx in range(data.shape[1]):
            if norm_methods[vidx] == "min-max":
                # min-max normalization
                data[:, vidx, ...] = (data[:, vidx, ...] - stats_2d[vidx, 0]) / (stats_2d[vidx, 1] - stats_2d[vidx, 0])
            elif norm_methods[vidx] == "max":
                # max normalization
                data[:, vidx, ...] = data[:, vidx, ...] / stats_2d[vidx, 1]
            elif norm_methods[vidx] == "std":
                # transform to standard normal
                data[:, vidx, ...] = (data[:, vidx, ...] - stats_2d[vidx, 0]) / stats_2d[vidx, 1]
            elif norm_methods[vidx] == "none":
                # leave data untouched
                pass
            else:
                LOGGER.error("Invalid normalization method %s for variable index %d ...", norm_methods[vidx], vidx)
                raise RuntimeError
        return data

    return _normalize_2d_era_data


def normalize_3d_era_data_wrapper(mu: np.ndarray, sd: np.ndarray) -> Callable:
    def _normalize_3d_era_data(data: np.ndarray) -> np.ndarray:
        LOGGER.debug("Worker %d produced 3D var data.shape: %s", os.getpid(), data.shape)
        # always standardize to N(0, 1)
        # 3D data.shape = (rollout, nvar, nlev, latlon)
        # assumes mu.shape and sd.shape align with the data dims
        return (data - mu) / sd

    return _normalize_3d_era_data
