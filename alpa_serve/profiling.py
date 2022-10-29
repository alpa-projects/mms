"""Profile the running time and memory usage of models"""
from collections import namedtuple
import dataclasses
from typing import List, Dict


# 3D parallel configuration
ParallelConfig = namedtuple("ParallelConfig", ("dp", "op", "pp"))


@dataclasses.dataclass
class ProfilingResult:
    """Store the profiling result of a model."""
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
                act_mem=None,
                weight_mem=None,
            )
        elif name == "alpa/bert-2.6b":
            return ProfilingResult(
                stage_latency={
                    ParallelConfig(1, 1, 1): {
                        1: [0.148],
                    },
                    ParallelConfig(1, 1, 2): {
                        1: [0.075, 0.076],
                    },
                },
                preprocess_cpu=0,
                postprocess_cpu=0,
                act_mem=None,
                weight_mem=None,
            )

        else:
            raise ValueError("Unsupported model: {name}")
