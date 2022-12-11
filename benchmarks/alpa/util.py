from functools import partial
import logging

from alpa_serve.simulator.executable import Executable

from benchmarks.alpa.bert_model import BertModel, bert_specs


def build_logger():
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_model_def(name, is_simulator, prof_database):
    result = prof_database.get(name)
    if result is None:
        raise ValueError(f"Invalid model name: {name}")

    if is_simulator:
        return partial(Executable, result)
    else:
        if name == "bert-1.3b":
            return partial(BertModel, bert_specs["1.3B"], result)
        elif name == "bert-2.6b":
            return partial(BertModel, bert_specs["2.6B"], result)
        elif name == "bert-6.7b":
            return partial(BertModel, bert_specs["6.7B"], result)
        elif name == "bert-103.5b":
            return partial(BertModel, bert_specs["103.5B"], result)
        else:
            raise ValueError(f"Invalid model name: {name}")
