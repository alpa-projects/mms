"""Profile the running time and memory usage of models"""
from collections import namedtuple
import csv
import dataclasses
import json
import pickle
from typing import List, Dict, Union


# 3D parallel configuration
ParallelConfig = namedtuple("ParallelConfig", ("dp", "op", "pp"))

@dataclasses.dataclass
class ProfilingResult:
    """Store the profiling result of a model."""
    model_name: str
    # The latency of each pipeline stage on GPUs.
    # type: Dict[parallel_config -> Dict[batch_size -> List[stage_latency]]]
    stage_latency: Dict
    # The latency of preprocess on CPU.
    preprocess_cpu: float
    # The latency of postprocess on CPU.
    postprocess_cpu: float
    # The activation memory in bytes.
    # Dict[batch_size -> mem]
    act_mem: Dict
    # The weight memory in bytes.
    weight_mem: float

    @staticmethod
    def load(name: str):
        if name == "alpa/bert-1.3b":
            return ProfilingResult(
                "alpa/bert-1.3b",
                stage_latency={
                    ParallelConfig(1, 1, 1): {
                        1: [0.099],
                    },
                    ParallelConfig(1, 1, 2): {
                        1: [0.051, 0.052],
                    },
                },
                preprocess_cpu=0,
                postprocess_cpu=0,
                act_mem={},
                weight_mem=0.0,
            )
        else:
            raise ValueError("Unsupported model: {name}")
    
    def add_result(self, parallel_config: ParallelConfig, batch_size: int, stage_latency: List[float], act_mem: float):
        """Add or overwrite the profiling results of a model."""
        if parallel_config not in self.stage_latency:
            self.stage_latency[parallel_config] = {}
        self.stage_latency[parallel_config][batch_size] = stage_latency
        if self.act_mem is None:
            self.act_mem = {}
        self.act_mem[batch_size] = act_mem


class ProfilingDatabase:
    """Store the profiling results of all the models"""
    def __init__(self, database_filename: str, new_database: bool = False):
        # The file that backs up the profiling results.
        self.database_filename = database_filename
        # Dict[model_name -> ProfilingResult]
        self.results = {}
        if not new_database:
            with open(database_filename, "rb") as f:
                self.results = pickle.load(f)

    def get(self, model_name: str) -> ProfilingResult:
        return self.results[model_name]
    
    def update(self, result: ProfilingResult):
        self.results[result.model_name] = result

    def _extract_data(self, row):
        """Extract the profiling results from a row of the profiling CSV file."""
        stage_latencies = list(map(float, row["StageLatencies(s)"].strip("[]").split()))
        # TODO: fix the activation memory
        act_mem = {}
        parallel_config = ParallelConfig(int(row["DP"]), int(row["OP"]), int(row["PP"]))
        return row["ModelName"], parallel_config, int(row["BS"]), stage_latencies, act_mem, float(row["Weights(GB)"])

    def update_from_csv(self, file_name: str):
        results = {}
        with open(file_name, "r") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                model_name, parallel_config, batch_size, stage_latencies, act_mem, weight_mem = self._extract_data(row)
                if model_name not in results:
                    results[model_name] = ProfilingResult(model_name, 
                                                          {parallel_config: {batch_size: stage_latencies}},
                                                          0, 0, act_mem, weight_mem)
                else:
                    results[model_name].add_result(parallel_config, batch_size, stage_latencies, act_mem)
        self.results = results
 
    def materialize(self):
        """Write the profiling results to the database file."""
        with open(self.database_filename, "wb") as f:
            pickle.dump(self.results, f)
    
database = ProfilingDatabase("profiling_results.pkl")