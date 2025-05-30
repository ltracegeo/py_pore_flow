import logging


DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Output destination
STDOUT = 1
FILE = 2
BOTH = 3


logger = logging.getLogger("py_pore_flow")


def configure(output=STDOUT, level=CRITICAL, timestamp=True, filename="py_pore_flow.log"):
    global logger

    logger.propagate = False
    if timestamp:
        formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s", datefmt="[%H:%M:%S]")
    else:
        formatter = logging.Formatter("%(levelname)s - %(message)s", datefmt="[%H:%M:%S]")

    if output & STDOUT:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if output & FILE:
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


def debug(message):
    global logger
    logger.debug(message)


def info(message):
    global logger
    logger.info(message)


def warning(message):
    global logger
    logger.warning(message)


def error(message):
    global logger
    logger.error(message)
