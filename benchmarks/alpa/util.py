from functools import partial
import logging

from alpa_serve.simulator.executable import Executable

from benchmarks.alpa.bert_model import BertModel, bert_specs


def build_logger():
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_model_def(name, is_simulator, prof_database=None):
    if is_simulator:
        result = prof_database.get(name)
        if result is None:
            raise ValueError(f"Invalid model name: {name}")
        else:
            return partial(Executable, result)
    else:
        if name == "bert-1.3b":
            return partial(BertModel, bert_specs["1.3B"])
        elif name == "bert-2.6b":
            return partial(BertModel, bert_specs["2.6B"])
        elif name == "bert-6.7b":
            return partial(BertModel, bert_specs["6.7B"])
        else:
            raise ValueError(f"Invalid model name: {name}")
