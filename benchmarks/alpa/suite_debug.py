from alpa_serve.simulator.controller import Controller
from alpa_serve.simulator.workload import Workload, PoissonProcess, GammaProcess
from alpa_serve.placement_policy.base_policy import ModelPlacement
from alpa_serve.profiling import ParallelConfig, ProfilingDatabase
from alpa_serve.util import ServingCase, GB

from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.equal_model_case import (EqualModelCase,
    get_equal_model_serving_case)


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
suite_debug["debug_pipeline_8x"] = debug_case(30, 60, "pipeline_8x")
suite_debug["debug_replicate_overloaded"] = debug_case(30, 30, "replicate")
suite_debug["debug_pipeline_overloaded"] = debug_case(30, 30, "pipeline_2x")
suite_debug["debug_invalid"] = debug_case(4, 30, "invalid")


def align_case_1():
    model_type = "bert-2.6b"
    rate = 10
    cv = 4
    duration = 30
    slo = 1 * 0.1

    def register_models(controller):
        is_simulator = isinstance(controller, Controller)

        controller.register_model.remote(
            "a", get_model_def(model_type, is_simulator, prof_database))

    def generate_workload(start=0):
        arrival_process = GammaProcess(rate, cv)
        w1 = arrival_process.generate_workload("a", start, duration,
                                               slo=slo, seed=0)
        w = w1
        return w

    def place_models(controller):
        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 8])
        controller.create_replica.remote(
            "a", group_id, [ParallelConfig(1, 2, 4)])

        model_placement = ModelPlacement(
            [ParallelConfig(1,2,4)], [[0]])
        controller.sync()
        return model_placement

    return ServingCase(register_models, generate_workload, place_models)


suite_debug["align_1"] = align_case_1()


def equal_model_case_1():
    num_devices = 8
    mem_budget = 14 * GB
    model_type = "bert-2.6b"
    num_models = 12
    total_rate = 35

    arrival_process = "gamma"
    rate_distribution = "power_law"
    arrival_process_kwargs = {"cv": 4}
    slo_scale = 1

    policy_name = "mp-search"

    duration = 60

    case = EqualModelCase(
        num_devices, mem_budget, model_type, num_models,
        total_rate, rate_distribution,
        arrival_process, arrival_process_kwargs,
        slo_scale, duration, policy_name)
    case = get_equal_model_serving_case(case)
    return case
