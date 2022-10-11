import logging


def build_logger():
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger

