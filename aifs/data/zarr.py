from zarr.core import Array


def read_zarr(uri: str) -> Array:
    import zarr

    if uri.startswith("http:") or uri.startswith("https:"):
        import s3fs

        bits = uri.split("/")
        url = "/".join(bits[:3])
        root = "/".join(bits[3:])
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": url})
        store = s3fs.S3Map(
            root=root,
            s3=fs,
            check=False,
        )
    else:
        store = zarr.DirectoryStore(uri)

    return zarr.open(store, "r")
