"""Inspect the profiling database."""
import pickle

import numpy as np

import alpa_serve
from alpa_serve.profiling import ParallelConfig

prof = pickle.load(open("profiling_result.pkl", "rb"))

parallel_configs = ParallelConfig(1, 1, 8)
bs = 1

#for model_name in ["bert-1.3b", "bert-2.6b", "bert-6.7b", "moe-1.3b", "moe-2.4b", "moe-7.1b"]:

for model_name in ["bert-1.3b", "bert-2.6b"]:
    for parallel_config in [ParallelConfig(1,1,1), ParallelConfig(1,1,8)]:
        base_latency = sum(prof[model_name].para_dict[ParallelConfig(1, 1, 1)].latency[bs])
        latency = prof[model_name].para_dict[parallel_config].latency[bs]
        print(f"Model: {model_name}, {parallel_config}, Latency: {sum(latency):.4f}, "
              f"Latency Overhead: {sum(latency) / base_latency:.2f}, "
              f"Throughput Overhead: {np.prod(parallel_config) * max(latency) / base_latency:.2f}")
