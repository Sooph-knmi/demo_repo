import ctypes
import dask


def trim_dask_worker_memory() -> int:
    """
    Manually trim Dask worker memory. This will forcefully release allocated but unutilized memory.
    This may help reduce total memory used per worker.
    See:
        https://distributed.dask.org/en/stable/worker-memory.html
    and
        https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
    """
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def init_dask_config(temp_dir: str) -> None:
    dask.config.set(
        {
            # temporary directory
            "temporary_directory": temp_dir,
            # this high initial guess tells the scheduler to spread tasks
            # "distributed.scheduler.unknown-task-duration": "10s",
            # worker memory management:
            # target - fraction of managed memory where we start spilling to disk
            "distributed.worker.memory.target": 0.85,
            # spill - fraction of process memory where we start spilling to disk
            "distributed.worker.memory.spill": 0.9,
            # pause - fraction of process memory at which we pause worker threads
            "distributed.worker.memory.pause": 0.95,
            # terminate - fraction of process memory at which we terminate the worker
            "distributed.worker.memory.terminate": False,
            "distributed.worker.use-file-locking": False,
            # MALLOC_TRIM_THRESHOLD_ - aggressively trim memory
            "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
        }
    )
