import zarr
from zarr.core import Array


def read_era_data(
    fname: str,
) -> Array:
    """Read ERA5 data from zarr file.

    Parameters
    ----------
    fname : str
        Filename of zarr file

    Returns
    -------
    Array
        Loaded zarr array
    """
    ds = zarr.open(fname, mode="r")
    # Should we use this?
    # ds = zarr.storage.LRUStoreCache(ds, 1024 * 1024 * 1024)
    if isinstance(ds, zarr.hierarchy.Group):
        return ds.data
    return ds
