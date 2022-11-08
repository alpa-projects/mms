from collections import namedtuple

from alpa_serve.simulator.controller import Controller
from alpa_serve.simulator.workload import Workload
from alpa_serve.profiling import ParallelConfig, ProfilingDatabase

from benchmarks.alpa.util import get_model_def


BenchmarkCase = namedtuple("BenchmarkCase",
    ("register_models", "generate_workload", "placement_policy"))


cases = {
}

prof_database = ProfilingDatabase("profiling_result.pkl", False)

def debug_case(placement):
    def register_models(controller):
        is_simulator = isinstance(controller, Controller)

        controller.register_model.remote(
            "a", get_model_def("bert-1.3b", is_simulator, prof_database))
        controller.register_model.remote(
            "b", get_model_def("bert-1.3b", is_simulator, prof_database))

    def generate_workload(start=0):
        w1 = Workload.gen_poisson("a", start, 4, 60, slo=0.5, seed=1)
        w2 = Workload.gen_poisson("b", start, 4, 60, slo=0.5, seed=2)
        w = w1 + w2
        return w

    def place_models(controller):
        if placement == "manual_1":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 1])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 1)])

            group_id = 1
            controller.create_mesh_group_manager.remote(group_id, [1, 1])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 1)])
        elif placement == "manual_2":
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
        elif placement == "manual_3":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 2])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 2)])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 2)])
        elif placement == "manual_4":
            group_id = 0
            controller.create_mesh_group_manager.remote(group_id, [1, 1])
            controller.create_replica.remote(
                "a", group_id, [ParallelConfig(1, 1, 1)])
            controller.create_replica.remote(
                "b", group_id, [ParallelConfig(1, 1, 1)])
        else:
            raise ValueError(f"Invalid placement: {placement}")

        controller.sync()

    return BenchmarkCase(register_models, generate_workload, place_models)


cases["debug_manual_1"] = debug_case("manual_1")
cases["debug_manual_2"] = debug_case("manual_2")
cases["debug_manual_3"] = debug_case("manual_3")
cases["debug_manual_4"] = debug_case("manual_4")
