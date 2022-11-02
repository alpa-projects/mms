from collections import namedtuple

from alpa_serve.simulator.controller import Controller
from alpa_serve.simulator.workload import Workload
from alpa_serve.profiling import ParallelConfig, load_test_prof_result
from alpa_serve.placement_policy import (SelectiveReplication,
    ModelParallelismPlacement, ClusterEnv, ModelData)
from alpa_serve.util import GB

from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.suite import BenchmarkCase
from benchmarks.alpa.simulate_one_case import simulate_one_case


def gen_case(slo, placement):
    cluster_env = ClusterEnv(num_devices=8, mem_budget=10*GB)
    num_models = 16

    slos = [slo] * num_models
    model_types = ["alpa/bert-1.3b"] * num_models
    average_rates = [10] * num_models
    duration = 60

    def register_models(controller):
        is_simulator = isinstance(controller, Controller)

        for i in range(num_models):
            controller.register_model.remote(
                f"m{i}", get_model_def(model_types[i], is_simulator))

    def generate_workload(start=0):
        w = Workload.empty()
        for i in range(num_models):
            w += Workload.gen_poisson(f"m{i}", start, average_rates[i],
                                      duration, slo=slos[i], seed=i)
        return w

    def place_models(controller):
        model_datas = []
        for i in range(num_models):
            model_datas.append(ModelData(f"m{i}", slos[i], average_rates[i],
                               load_test_prof_result(model_types[i])))

        if placement == "sr":
            policy = SelectiveReplication()
        elif placement == "mp":
            policy = ModelParallelismPlacement()
        else:
            raise ValueError(f"Invalid placement policy: {placement}")

        policy.place_models(controller, model_datas, cluster_env)

    return BenchmarkCase(register_models, generate_workload, place_models)


if __name__ == "__main__":
    stats = simulate_one_case(gen_case(0.2, "sr"))
    print(stats.average_goodput)
