"""Profile the running time and memory usage of models"""
from collections import namedtuple
import csv
import dataclasses
import json
import pickle
from typing import List, Dict, Union, Any

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
    # Metadata for parallel strategy
    metadata: Any = None

    def add_result(self, batch_size: int, latency: List[float], act_mem: List[float], weight_mem: List[float], metadata: Any = None):
        if batch_size not in self.latency:
            self.latency[batch_size] = latency
            self.act_mem[batch_size] = act_mem
        else:
            new_max_latency = max(latency)
            old_max_latency = max(self.latency[batch_size])
            if new_max_latency < old_max_latency:
                self.latency[batch_size] = latency
                self.act_mem[batch_size] = act_mem
                self.weight_mem = weight_mem
                self.metadata = metadata


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
                   stage_latency: List[float], act_mem: List[float], weight_mem: List[float],
                   metadata: Any = None):
        """Add or overwrite the profiling results of a model."""
        if parallel_config not in self.para_dict:
            self.para_dict[parallel_config] = LatencyMemData(
                latency={batch_size: stage_latency},
                act_mem={batch_size: act_mem},
                weight_mem=weight_mem,
                metadata=metadata)
        else:
            self.para_dict[parallel_config].add_result(batch_size, stage_latency,
                                                       act_mem, weight_mem, metadata)


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
        return self.results.get(model_name)

    def update(self, result: ProfilingResult):
        self.results[result.model_name] = result

    def _extract_data(self, row):
        """Extract the profiling results from a row of the profiling CSV file."""
        stage_latencies = list(map(float, row["StageLatencies(s)"].strip("[]").split(",")))
        weight_mem = list(map(float, row["StageWeights(B)"].strip("[]").split(",")))
        peak_mem = list(map(float, row["StagePeakMem(B)"].strip("[]").split(",")))
        act_mem = [peak_mem - weight_mem for peak_mem, weight_mem in zip(peak_mem, weight_mem)]
        assert min(act_mem) > 0, "negative activation memory"
        parallel_config = ParallelConfig(int(row["DP"]), int(row["OP"]), int(row["PP"]))
        return row["ModelName"], parallel_config, int(row["BS"]), stage_latencies, weight_mem, act_mem

    def update_from_csv(self, file_name: str):
        # Add head if there is no head
        missing_head = False
        with open(file_name, "r") as f:
            l = f.readline()
            if l[0] != 'M':
                lines = [l] + f.readlines()
                missing_head = True

        if missing_head:
            heads = [
                "ModelName", "BS", "#Microbatch", "DP", "OP", "PP", "#GPU",
                "MeanTime(s)", "StdTime(s)", "TFLOPs", "StageWeights(B)",
                "StagePeakMem(B)", "StageLatencies(s)"
            ]
            lines = ["\t".join(heads) + "\n"] + lines
            with open(file_name, "w") as f:
                f.writelines(lines)

        # read lines
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

    def _extract_auto_data(self, row):
        """Extract the profiling results from a row of the profiling CSV file."""
        stage_latencies = list(map(float, row["StageLatencies(s)"].strip("[]").split(", ")))
        weight_mem = list(map(float, row["StageWeights(B)"].strip("[]").split(", ")))
        peak_mem = list(map(float, row["StagePeakMem(B)"].strip("[]").split(", ")))
        act_mem = [peak_mem - weight_mem for peak_mem, weight_mem in zip(peak_mem, weight_mem)]
        assert min(act_mem) > 0, "negative activation memory"
        metadata = eval(row["Metadata"])
        pp = len(metadata["submesh_shapes"])
        op = metadata["submesh_shapes"][0][1]
        parallel_config = ParallelConfig(1, op, pp)
        return row["ModelName"], parallel_config, int(row["BS"]), stage_latencies, weight_mem, act_mem, metadata

    def update_from_auto_csv(self, file_name: str):
        fieldnames = [
                "ModelName", "BS", "#Microbatch", "ParallelArgs", "MeanTime(s)",
                "StdTime(s)", "TFLOPs", "StageWeights(B)", "StagePeakMem(B)",
                "StageLatencies(s)", "Metadata", "TimeStamp"
            ]
        with open(file_name, "r") as f:
            reader = csv.DictReader(f, fieldnames=fieldnames, delimiter="\t")
            for row in reader:
                model_name, parallel_config, batch_size, stage_latencies, weight_mem, act_mem, metadata = self._extract_auto_data(row)
                print(model_name, parallel_config, batch_size, stage_latencies, weight_mem, act_mem, metadata)
                if model_name not in self.results:
                    self.results[model_name] = ProfilingResult(
                        model_name,
                        {
                            parallel_config: LatencyMemData(
                                latency={   # Dict[batch_size -> List[stage_latency]]
                                    batch_size: stage_latencies,
                                },
                                act_mem={   # Dict[batch_size -> List[stage_act_mem]]
                                    batch_size: act_mem,
                                },
                                weight_mem=weight_mem, # List[stage_weight_mem]
                                metadata=metadata,
                            )
                        },
                        preprocess_cpu=0.0,
                        postprocess_cpu=0.0
                    )
                else:
                    self.results[model_name].add_result(parallel_config, batch_size, stage_latencies, act_mem, weight_mem, metadata)


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
                ParallelConfig(1, 1, 4): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.028, 0.027, 0.027, 0.029],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO", "TODO", "TODO", "TODO"],
                    },
                    weight_mem=[  # List[stage_weight_mem]
                        0.65*GB, 0.65*GB, 0.65*GB, 0.65*GB,
                    ],
                ),
                ParallelConfig(1, 1, 8): LatencyMemData(
                    latency={     # Dict[batch_size -> List[stage_latency]]
                        1: [0.015, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.016],
                    },
                    act_mem={     # Dict[batch_size -> List[stage_act_mem]]
                        1: ["TODO"] * 8,
                    },
                    weight_mem=[0.325*GB] * 8,
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
        raise ValueError(f"Unsupported model: {name}")

# database = ProfilingDatabase("profiling_result.pkl", False)
# bert_1_3b = database.get("bert-1.3b")
# bert_2_6b = database.get("bert-2.6b")
# bert_6_7b = database.get("bert-6.7b")
# print(bert_1_3b.para_dict[ParallelConfig(1,4,4)].latency[8])
# print(bert_2_6b.para_dict[ParallelConfig(1,4,4)].latency[8])
# print(bert_6_7b.para_dict[ParallelConfig(1,4,4)].latency[8])
