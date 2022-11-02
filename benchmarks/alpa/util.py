from functools import partial
import logging

from alpa_serve.profiling import load_test_prof_result
from alpa_serve.simulator.executable import Executable

from benchmarks.alpa.bert_model import BertModel, bert_specs


def build_logger():
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_model_def(name, is_simulator):
    if is_simulator:
        if name == "alpa/bert-1.3b":
            return partial(Executable, load_test_prof_result("alpa/bert-1.3b"))
        elif name == "alpa/bert-2.6b":
            return partial(Executable, load_test_prof_result("alpa/bert-2.6b"))
        else:
            raise ValueError(f"Invalid model name: {name}")
    else:
        if name == "alpa/bert-1.3b":
            return partial(BertModel, bert_specs["1.3B"])
        elif name == "alpa/bert-2.6b":
            return partial(BertModel, bert_specs["2.6B"])
        else:
            raise ValueError(f"Invalid model name: {name}")
