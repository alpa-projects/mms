import logging

def build_logger():
    logger = logging.getLogger("alpa_serve_bench")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger

