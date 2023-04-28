from time import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from aifs.utils.config import YAMLConfig
from aifs.data.era_datamodule import ERA5DataModule
from aifs.utils.logger import get_logger
from aifs.train.utils import get_args

LOGGER = get_logger(__name__)


def dltest(config: YAMLConfig) -> None:
    """
    Train entry point.
    Args:
        config: job configuration
    """
    # create data module (data loaders and data sets)
    dmod = ERA5DataModule(config)

    dl: DataLoader = dmod.train_dataloader()

    start_time = time()
    for _ in tqdm(dl):
        pass
    end_time = time()

    LOGGER.debug("Elapsed time (one training epoch): %d s", (start_time - end_time))
    LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Entry point for the data loader test."""
    args = get_args()
    config = YAMLConfig(args.config)
    dltest(config)
