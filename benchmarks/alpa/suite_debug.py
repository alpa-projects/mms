from alpa_serve.simulator.controller import Controller
from alpa_serve.simulator.workload import Workload, PoissonProcess
from alpa_serve.placement_policy.base_policy import ModelPlacement
from alpa_serve.profiling import ParallelConfig, ProfilingDatabase
from alpa_serve.util import ServingCase

from benchmarks.alpa.util import get_model_def


suite_debug = {
}

prof_database = ProfilingDatabase("profiling_result.pkl", False)

def debug_case(per_model_rate, duration, placement):
    def register_models(controller):
        is_simulator = isinstance(controller, Controller)

        controller.register_model.remote(
            "a", get_model_def("bert-1.3b", is_simulator, prof_database))
        controller.register_model.remote(
            "b", get_model_def("bert-1.3b", is_simulator, prof_database))

    def generate_workload(start=0):
        arrival_process = PoissonProcess(per_model_rate)
        w1 = arrival_process.generate_workload("a", start, duration, slo=0.5, seed=1)
        w2 = arrival_process.generate_workload("b", start, duration, slo=0.5, seed=2)
        w = w1 + w2
        return w

    def place_models(controller):
        if placement == "replicate":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 1])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 1)])

            group_id = 1
            controller.create_mesh_group_manager.remote(group_id, [1, 1])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 1)])

            model_placement = ModelPlacement(
                [ParallelConfig(1,1,1), ParallelConfig(1,1,1)], [[0], [1]])
        elif placement == "replicate_2x":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 1])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 1)])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 1)])

            group_id = 1
            controller.create_mesh_group_manager.remote(group_id, [1, 1])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 1)])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 1)])

            model_placement = ModelPlacement(
                [ParallelConfig(1,1,1), ParallelConfig(1,1,1)], [[0,1], [0,1]])
        elif placement == "pipeline_2x":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 2])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 2)])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 2)])

            model_placement = ModelPlacement([ParallelConfig(1,1,2)], [[0, 1]])
        elif placement == "pipeline_8x":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 8])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 8)])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 8)])

            model_placement = ModelPlacement([ParallelConfig(1,1,8)], [[0, 1]])
        elif placement == "invalid":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 1])

            group_id = 1
            controller.create_mesh_group_manager.remote(group_id, [1, 1])

            model_placement = ModelPlacement([], [])
        else:
            raise ValueError(f"Invalid placement: {placement}")

        controller.sync()
        return model_placement

    return ServingCase(register_models, generate_workload, place_models)


suite_debug["debug_replicate"] = debug_case(4, 60, "replicate")
suite_debug["debug_replicate_2x"] = debug_case(4, 60, "replicate_2x")
suite_debug["debug_pipeline_2x"] = debug_case(4, 60, "pipeline_2x")
suite_debug["debug_replicate_overloaded"] = debug_case(30, 30, "replicate")
suite_debug["debug_pipeline_overloaded"] = debug_case(30, 30, "pipeline_2x")
suite_debug["debug_invalid"] = debug_case(4, 30, "invalid")
