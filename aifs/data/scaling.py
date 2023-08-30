import numpy as np


def pressure_level(plev) -> np.ndarray:
    """Convert pressure levels to PyTorch Lightning scaling."""
    return np.array(plev) / 1000
