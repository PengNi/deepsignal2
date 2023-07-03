import sys
import logging


CONSOLE = logging.StreamHandler()
CONSOLE.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s"
)
CONSOLE.setFormatter(formatter)
ROOT_LOGGER = logging.getLogger("Remora")
ROOT_LOGGER.setLevel(logging.DEBUG)
ROOT_LOGGER.addHandler(CONSOLE)


def init_logger(log_fn=None, quiet=False):
    """Prepare logging output.

    Args:
        log_fn (str): Path to logging output file. All logging messages,
            including debug level, will be output to this file.
        quiet (bool): Set console logging level to warning. Default info.
    """
    log_fp = None
    if log_fn is not None:
        log_fp = logging.FileHandler(log_fn, mode="a")
        log_fp.setLevel(logging.DEBUG)
        log_fp.setFormatter(formatter)
    if log_fp is not None:
        ROOT_LOGGER.addHandler(log_fp)
    if quiet:
        CONSOLE.setLevel(logging.WARNING)
    logging.getLogger("Remora").debug(f'Command: """{" ".join(sys.argv)}"""')


def get_logger(module_name="deepsignal"):
    return logging.getLogger(module_name)
