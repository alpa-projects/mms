import logging


def build_logger():
    logger = logging.getLogger("alpa_serve")
    logger.setLevel(logging.INFO)
    return logger
