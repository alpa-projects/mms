""" A pipelined executable. """
import dataclasses
from typing import List, Dict

from alpa_serve.placement_policy import ParallelConfig
from alpa_serve.simulator.cluster import VirtualMesh
from alpa_serve.simulator.event_loop import wait_multi_stream


@dataclasses.dataclass
class ProfilingResult:
    """Store the profiling result of an executable"""
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
                        1: [0.10],
                        2: [0.15],
                    },
                    ParallelConfig(1, 1, 2): {
                        1: [0.050, 0.050],
                        2: [0.075, 0.075],
                    },
                },
                preprocess_cpu=0,
                postprocess_cpu=0,
                act_mem=None,
                weight_mem=None,
            )
        else:
            raise ValueError("Unsupported model: {name}")


class Executable:
    def __init__(self,
                 profiling_result: ProfilingResult,
                 parallel_config: ParallelConfig,
                 virtual_mesh: VirtualMesh):
        self.profile = profiling_result
        self.parallel_config = parallel_config
        self.stage_latency = profiling_result.stage_latency[parallel_config]

        # launch or connect to a mesh group
        submesh_shapes = (
            (parallel_config.dp, parallel_config.op),) * parallel_config.pp
        if virtual_mesh.launched_mesh_group:
            assert submesh_shape == virtual_mesh.submesh_shapes
            mesh_group = virtual_mesh.launched_mesh_group
        else:
            mesh_group = virtual_mesh.launch_mesh_group(submesh_shapes)

        self.mesh_group = mesh_group

    async def execute(self, batch_size: int):
        latencies = self.stage_latency[batch_size]
        for mesh, latency in zip(self.mesh_group.meshes, latencies):

            streams = [g.stream_name for g in mesh.gpus]
            durations = [latency] * len(streams)
            await wait_multi_stream(streams, durations)
