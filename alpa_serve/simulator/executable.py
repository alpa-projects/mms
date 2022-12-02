"""A pipeline executable."""
from alpa_serve.profiling import ParallelConfig, ProfilingResult
from alpa_serve.simulator.cluster import VirtualMesh
from alpa_serve.simulator.event_loop import clock, timed_coroutine, wait_stream, wait_multi_stream, sleep


class Executable:
    def __init__(self,
                 profiling_result: ProfilingResult,
                 parallel_config: ParallelConfig,
                 virtual_mesh: VirtualMesh):
        self.profile = profiling_result
        self.parallel_config = parallel_config
        self.latency_mem = profiling_result.para_dict[parallel_config]

        # launch or connect to a mesh group
        submesh_shapes = (
            (parallel_config.dp, parallel_config.op),) * parallel_config.pp
        if virtual_mesh.launched_mesh_group:
            assert submesh_shapes == virtual_mesh.submesh_shapes
            mesh_group = virtual_mesh.launched_mesh_group
        else:
            mesh_group = virtual_mesh.launch_mesh_group(submesh_shapes)

        self.mesh_group = mesh_group

    def get_latency_dict(self):
        return self.latency_mem.latency

    @timed_coroutine
    async def handle_request(self, request):
        request.time_stamp["d"] = clock()
        batch_size = 1

        stage_latency = self.latency_mem.latency[batch_size]
        for mesh, latency in zip(self.mesh_group.meshes, stage_latency):
            # SPMD version
            stream = mesh.gpus[0].stream_name
            await wait_stream(stream, latency)

            # More accurate version
            #streams = [g.stream_name for g in mesh.gpus]
            #durations = [latency] * len(streams)
            #await wait_multi_stream(streams, durations)
        request.time_stamp["e"] = clock()
        return True
