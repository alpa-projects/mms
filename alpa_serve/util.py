"""Common utilities."""
import logging


def build_logger(name="alpa_serve"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
