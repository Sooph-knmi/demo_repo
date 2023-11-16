import logging
import os
import sys
import time


def get_code_logger(name: str, debug: bool = True) -> logging.Logger:
    """Returns a logger with a custom level and format.

    We use ISO8601 timestamps and UTC times.

    Parameters
    ----------
    name : str
        Name of logger object
    debug : bool, optional
        set logging level to logging.DEBUG; else set to logging.INFO, by default True

    Returns
    -------
    logging.Logger
        Logger object
    """
    # create logger object
    logger = logging.getLogger(name=name)

    if not logger.hasHandlers():
        # logging level
        level = logging.DEBUG if debug else logging.INFO
        # custom format field
        global_rank = int(os.environ.get("SLURM_PROCID", "0"))
        extra_field = {"global_rank": f"global_rank {global_rank:03d}"}

        # logging format
        datefmt = "%Y-%m-%dT%H:%M:%SZ"
        msgfmt = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName).30s] [%(global_rank)s] [%(levelname)s] %(message)s"
        # handler object
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(msgfmt, datefmt=datefmt)
        # record UTC time
        setattr(formatter, "converter", time.gmtime)
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger = logging.LoggerAdapter(logger, extra_field)
    return logger
