import zarr
from zarr.core import Array


def read_era_data(
    fname: str,
) -> Array:
    ds = zarr.open(fname, mode="r")
    # Should we use this?
    # ds = zarr.storage.LRUStoreCache(ds, 1024 * 1024 * 1024)
    return ds