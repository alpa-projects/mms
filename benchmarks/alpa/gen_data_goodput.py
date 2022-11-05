import argparse
from collections import namedtuple

import ray

from alpa_serve.simulator.controller import Controller
from alpa_serve.simulator.workload import Workload
from alpa_serve.profiling import ParallelConfig, load_test_prof_result, ProfilingDatabase
from alpa_serve.placement_policy import (SelectiveReplication,
    ModelParallelismPlacement, ClusterEnv, ModelData)
from alpa_serve.util import GB, write_tsv

from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.suite import BenchmarkCase
from benchmarks.alpa.simulate_one_case import simulate_one_case


def gen_case(slo, placement, prof_database,
             num_devices, num_models, mem_budget,
             average_rate, cv, duration):
    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget)
    num_models = num_models

    slos = [slo] * num_models
    model_types = ["bert-1.3b"] * num_models
    average_rates = [average_rate] * num_models
    cvs = [cv] * num_models
    duration = duration

    def register_models(controller):
        is_simulator = isinstance(controller, Controller)

        for i in range(num_models):
            controller.register_model.remote(
                f"m{i}", get_model_def(model_types[i], is_simulator, prof_database))

    def generate_workload(start=0):
        w = Workload.empty()
        for i in range(num_models):
            w += Workload.gen_gamma(f"m{i}", start, average_rates[i], cv=cvs[i],
                                    duration=duration, slo=slos[i], seed=i)
        return w

    def place_models(controller):
        model_datas = []
        for i in range(num_models):
            model_datas.append(ModelData(f"m{i}", slos[i], average_rates[i],
                               prof_database.get(model_types[i])))

        if placement == "sr":
            policy = SelectiveReplication()
        elif placement == "mp":
            policy = ModelParallelismPlacement()
        else:
            raise ValueError(f"Invalid placement policy: {placement}")

        policy.place_models(controller, model_datas, cluster_env)

    return BenchmarkCase(register_models, generate_workload, place_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_goodput.tsv")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    prof_database = ProfilingDatabase("profiling_result.pkl")

    policies = ["sr"]
    slos = [0.1, 0.15, 0.2, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0]
    goodputs = []

    heads = ["exp_name", "policy", "slo", "goodput"]

    if args.parallel:
        ray.init(address="auto")
        simulate_one_case = ray.remote(num_cpus=2)(simulate_one_case).remote

    stats_res = {}
    for policy in policies:
        for slo in slos:
            stats_res[(policy, slo)] = simulate_one_case(
                gen_case(slo, policy,
                         prof_database=prof_database,
                         num_devices=8, num_models=16, mem_budget=10*GB,
                         average_rate=5, cv=4, duration=100))

    for policy in policies:
        for slo in slos:
            if args.parallel:
                stats = ray.get(stats_res[(policy, slo)])
            else:
                stats = stats_res[(policy, slo)]
            goodput = stats.average_goodput
            Workload.print_stats(stats)
            goodputs.append(goodput)

            values = [args.exp_name, policy, slo, goodput]
            write_tsv(heads, values, args.output)
