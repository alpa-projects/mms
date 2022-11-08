import argparse
from collections.abc import Iterable

import numpy as np
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
from benchmarks.alpa.run_one_case import run_one_case


def generate_gamma_workloads(model_names, average_rate, cv, duration,
                             slo, start=0):
    """Generate a workload where the requests to each model follows a gamma
    process, where the gap between the requests follows a Gamma distribution.

    Args:
        model_names (list[str]): Names of the models.
        average_rate (float): The average number of requests per second to each
            model.
        cv (float): The coefficient of variation of the gap between the
            requests. Higher cv leads to a more bursty workload. cv == 1 is
            a Poisson process.
        duration (float): The duration of the workload.
        slo (float): The service level objective of each model.
        start (Optional[float]): The start time of the workload.
    """

    w = Workload.empty()
    for i, model_name in enumerate(model_names):
        w += Workload.gen_gamma(model_name, start, average_rate,
                                cv=cv, duration=duration, slo=slo,
                                seed=i)
    return w


def gen_uniform_mmpp_workloads(model_names, num_requests, state_durations,
                               state_request_rates, slo, start=0):
    """Generate a workload where the requests to each model follows a Markov
    Modulated Poisson Process (MMPP), where the transition probability among
    the states of the Markov chain is uniform across all states.

    Args:
        model_names (list[str]): Names of the models.
        num_requests (int): The total number of requests to generate.
        stream_durations (list[float]): The duration of each stream.
        stream_request_rates (list[float]): The request rate of each stream.
    """
    w = Workload.empty()
    for i, model_name in enumerate(model_names):
        w += Workload.gen_uniform_mmpp(model_name, start, num_requests,
                                       state_durations, state_request_rates,
                                       slo=slo, seed=i)
    return w


def register_models(controller, model_names, model_types, prof_database):
    is_simulator = isinstance(controller, Controller)

    for model_name, model_type in zip(model_names, model_types):
        controller.register_model.remote(
            model_name, get_model_def(model_type, is_simulator,
                                      prof_database))


def place_models(controller, cluster_env, placement, model_names, model_types,
                 average_rates, slos, prof_database):
    num_models = len(model_names)
    model_datas = []
    for i in range(num_models):
        model_datas.append(ModelData(model_names[i], slos[i], average_rates[i],
                           prof_database.get(model_types[i])))

    if placement == "sr":
        policy = SelectiveReplication(verbose=True)
    elif placement == "mp":
        policy = ModelParallelismPlacement(verbose=True)
    else:
        raise ValueError(f"Invalid placement policy: {placement}")

    policy.place_models(controller, model_datas, cluster_env)
    return policy


def gen_gamma_case(slo, placement, prof_database,
                   num_devices, num_models, mem_budget,
                   average_rate, cv, duration):
    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget)
    num_models = num_models

    slos = [slo] * num_models
    model_names = [f"m{i}" for i in range(num_models)]
    model_types = ["bert-1.3b"] * num_models
    average_rates = [average_rate] * num_models

    def register_models_(controller):
        return register_models(
            controller, model_names, model_types, prof_database)

    def generate_workload_(start=0):
        return generate_gamma_workloads(model_names, average_rate, cv,
                                        duration, slo, start)

    def place_models_(controller):
        return place_models(controller, cluster_env, placement, model_names,
                            model_types, average_rates, slos, prof_database)

    return BenchmarkCase(register_models_, generate_workload_, place_models_)


def gen_uniform_mmpp_case(slo, placement, prof_database,
                          num_devices, num_models, mem_budget,
                          state_durations, state_request_rates, num_requests):
    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget)
    num_models = num_models

    slos = [slo] * num_models
    model_names = [f"m{i}" for i in range(num_models)]
    model_types = ["bert-1.3b"] * num_models
    state_durations = np.array(state_durations)
    state_request_rates = np.array(state_request_rates)
    average_rate = (np.sum(state_request_rates * state_durations)
                    / np.sum(state_durations))
    average_rates = [average_rate] * num_models

    def register_models_(controller):
        return register_models(
            controller, model_names, model_types, prof_database)

    def generate_workload_(start=0):
        return gen_uniform_mmpp_workloads(model_names, num_requests,
                                          state_durations, state_request_rates,
                                          slo, start)

    def place_models_(controller):
        return place_models(controller, cluster_env, placement, model_names,
                            model_types, average_rates, slos, prof_database)

    return BenchmarkCase(register_models_, generate_workload_, place_models_)


def run_experiment_slos(policies, slos, cases, exp_name="default",
                        output_file=None, mode="simulate", parallel=False):
    if mode == "simulate":
        if parallel:
            ray.init(address="auto")
            run_one_case_ = ray.remote(num_cpus=2)(simulate_one_case).remote
        else:
            run_one_case_ = simulate_one_case
    else:
        ray.init(address="auto")
        run_one_case_ = run_one_case

    stats_res = {}
    for policy in policies:
        for slo in slos:
            stats_res[(policy, slo)] = run_one_case_(cases[(policy, slo)])
    heads = ["exp_name", "policy", "slo", "goodput", "placement", "mode"]
    results = []
    for policy in policies:
        for slo in slos:
            if parallel:
                stats, placement_policy = ray.get(stats_res[(policy, slo)])
            else:
                stats, placement_policy = stats_res[(policy, slo)]
            goodput = stats.average_goodput
            Workload.print_stats(stats)

            values = [exp_name, policy, slo, f"{goodput:.3f}", str(placement_policy),
                      mode]
            results.append(values)
            if output_file is not None:
                write_tsv(heads, values, output_file)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_goodput.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()

    prof_database = ProfilingDatabase("profiling_result.pkl")

    policies = ["mp"]
    slos = [0.2]
    cases = {}
    for policy in policies:
        for slo in slos:
            cases[(policy, slo)] = gen_gamma_case(
                slo, policy, prof_database,
                num_devices=8, num_models=16, mem_budget=10*GB,
                average_rate=4, cv=4, duration=100)

    run_experiment_slos(policies, slos, cases,
                        exp_name=args.exp_name,
                        output_file=args.output,
                        mode=args.mode,
                        parallel=args.parallel)
