"""Profile the running time and memory usage of models"""
from collections import namedtuple
import csv
import dataclasses
import json
import pickle
from typing import List, Dict, Union

from alpa_serve.util import GB


# 3D parallel configuration
# (data parallel, operator parallel, pipeline parallel)
ParallelConfig = namedtuple("ParallelConfig", ("dp", "op", "pp"))

@dataclasses.dataclass
class LatencyMemData:
    # The latency of each stage
    # Type: Dict[batch_size -> List[stage_latency]]
    latency: Dict
    # The activation memory of each stage
    # Type: Dict[batch_size -> List[stage_act_mem]]
    act_mem: Dict
    # The weight memory of each stage
    # Type: List[stage_weight_mem]
    weight_mem: List

    def add_result(self, batch_size: int, latency: List[float], act_mem: List[float]):
        self.latency[batch_size] = latency
        self.act_mem[batch_size] = act_mem


@dataclasses.dataclass
class ProfilingResult:
    """Store the profiling result of a model."""
    model_name: str
    # The latency and memory usage of each pipeline stage on GPUs.
    # type: Dict[parallel_config -> latency_mem]
    para_dict: Dict
    # The latency of preprocess on CPU.
    preprocess_cpu: float
    # The latency of postprocess on CPU.
    postprocess_cpu: float

    def add_result(self, parallel_config: ParallelConfig, batch_size: int,
                   stage_latency: List[float], act_mem: List[float], weight_mem: List[float]):
        """Add or overwrite the profiling results of a model."""
        if parallel_config not in self.para_dict:
            self.para_dict[parallel_config] = LatencyMemData(
                latency={batch_size: stage_latency},
                act_mem={batch_size: act_mem},
                weight_mem=weight_mem)
        else:
            self.para_dict[parallel_config].add_result(batch_size, stage_latency, act_mem)


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
        weight_mem = list(map(float, row["StageWeights(GB)"].strip("[]").split()))
        peak_mem = list(map(float, row["StagePeakMem(GB)"].strip("[]").split()))
        # TODO: fix the activation memory
        act_mem = [peak_mem - weight_mem for peak_mem, weight_mem in zip(peak_mem, weight_mem)]
        parallel_config = ParallelConfig(int(row["DP"]), int(row["OP"]), int(row["PP"]))
        return row["ModelName"], parallel_config, int(row["BS"]), stage_latencies, weight_mem, act_mem

    def update_from_csv(self, file_name: str):
        results = {}
        with open(file_name, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                model_name, parallel_config, batch_size, stage_latencies, weight_mem, act_mem = self._extract_data(row)
                if model_name not in results:
                    results[model_name] = ProfilingResult(
                                                            model_name,
                                                            {
                                                                parallel_config: LatencyMemData(
                                                                    latency={   # Dict[batch_size -> List[stage_latency]]
                                                                        batch_size: stage_latencies,
                                                                    },
                                                                    act_mem={   # Dict[batch_size -> List[stage_act_mem]]
                                                                        batch_size: act_mem,
                                                                    },
                                                                    weight_mem=weight_mem # List[stage_weight_mem]
                                                                )
                                                            },
                                                            preprocess_cpu=0.0,
                                                            postprocess_cpu=0.0
                                                        )
                else:
                    results[model_name].add_result(parallel_config, batch_size, stage_latencies, act_mem, weight_mem)
        self.results.update(results)

    def materialize(self):
        """Write the profiling results to the database file."""
        with open(self.database_filename, "wb") as f:
            pickle.dump(self.results, f)


def load_test_prof_result(name: str):
    """Load pre-defined profiling results for testing."""
    if name == "alpa/bert-1.3b":
        return ProfilingResult("alpa/bert-1.3b",
            para_dict={
                ParallelConfig(1, 1, 1): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.099],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        2.6*GB,
                    ],
                ),
                ParallelConfig(1, 1, 2): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.051, 0.052],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO", "TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        1.3*GB, 1.3*GB,
                    ],
                ),
            },
            preprocess_cpu=0,
            postprocess_cpu=0,
        )
    elif name == "alpa/bert-2.6b":
        return ProfilingResult("alpa/bert-2.6b",
            para_dict={
                ParallelConfig(1, 1, 1): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.148],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        5.2*GB,
                    ],
                ),
                ParallelConfig(1, 1, 2): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.075, 0.076],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO", "TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        2.6*GB, 2.6*GB,
                    ],
                ),
            },
            preprocess_cpu=0,
            postprocess_cpu=0,
        )
    elif name == "test-2GB-100ms":
        return ProfilingResult("test-2GB-100ms",
            para_dict={
                ParallelConfig(1, 1, 1): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.100],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        2*GB,
                    ],
                ),
                ParallelConfig(1, 1, 2): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.051, 0.051],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO", "TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        1*GB, 1*GB
                    ],
                ),
                ParallelConfig(1, 1, 4): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.026, 0.026, 0.026, 0.026],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO", "TODO", "TODO", "TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        0.5*GB, 0.5*GB, 0.5*GB, 0.5*GB
                    ],
                ),
            },
            preprocess_cpu=0,
            postprocess_cpu=0,
        )
    elif name == "test-4GB-150ms":
        return ProfilingResult("test-4GB-150ms",
            para_dict={
                ParallelConfig(1, 1, 1): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.150],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        4*GB,
                    ],
                ),
                ParallelConfig(1, 1, 2): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.076, 0.076],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO", "TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        2*GB, 2*GB
                    ],
                ),
                ParallelConfig(1, 1, 4): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.039, 0.039, 0.039, 0.039],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO", "TODO", "TODO", "TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        1*GB, 1*GB, 1*GB, 1*GB
                    ],
                ),
            },
            preprocess_cpu=0,
            postprocess_cpu=0,
        )
    else:
        raise ValueError("Unsupported model: {name}")

# database = ProfilingDatabase("profiling_result.pkl", False)
# bert_1_3b = database.get("bert-1.3b")
# print(bert_1_3b.para_dict[ParallelConfig(1, 4, 4)].latency[8])