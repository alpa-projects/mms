"""Inspect the profiling database."""
import pickle

import alpa_serve
from alpa_serve.profiling import ParallelConfig

prof = pickle.load(open("profiling_result.pkl", "rb"))

model_name = "bert-1.3b"
parallel_config = ParallelConfig(1, 1, 1)
bs = 1

for model_name in ["bert-1.3b", "bert-2.6b", "bert-6.7b", "moe-1.3b", "moe-2.4b", "moe-7.1b"]:
    print(prof[model_name].para_dict[parallel_config].latency[bs])
