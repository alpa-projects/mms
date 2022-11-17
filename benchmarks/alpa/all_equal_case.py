import os
from collections import namedtuple, defaultdict

import ray

from alpa_serve.simulator.controller import Controller, simulate_one_case
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.placement_policy import (ClusterEnv, ModelData,
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    ModelParallelismILP, ModelParallelismGreedy, ModelParallelismSearch)
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.util import GB, write_tsv, ServingCase

from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.run_one_case import run_one_case


# A case where all models and all request distributions are the same.
AllEqualCase = namedtuple("AllEqualCase", [
    "num_devices", "mem_budget",
    "model_type", "num_models", "arrival_process",
    "slo", "duration", "policy_name"])


def get_all_equal_serving_case(case, prof_database=None):
    if prof_database is None:
        prof_database = ProfilingDatabase("profiling_result.pkl")

    (num_devices, mem_budget, model_type, num_models,
     arrival_process, slo, duration, policy_name) = case

    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget)
    num_models = num_models

    slos = [slo] * num_models
    model_names = [f"m{i}" for i in range(num_models)]
    model_types = [model_type] * num_models

    rates = [arrival_process.rate()] * num_models
    cvs = [arrival_process.cv()] * num_models

    def register_models(controller):
        is_simulator = isinstance(controller, Controller)

        for model_name, model_type in zip(model_names, model_types):
            controller.register_model.remote(
                model_name, get_model_def(model_type, is_simulator,
                                          prof_database))

    def generate_workload(start=0):
        w = Workload.empty()
        for i, model_name in enumerate(model_names):
            w += arrival_process.generate_workload(model_name, start, duration,
                                                   slo=slos[i], seed=i)
        return w

    def place_models(controller):
        num_models = len(model_names)
        model_datas = []
        for i in range(num_models):
            model_datas.append(ModelData(model_names[i], slos[i], rates[i], cvs[i],
                               prof_database.get(model_types[i])))

        if policy_name == "sr-ilp":
            policy = SelectiveReplicationILP(verbose=1)
        elif policy_name == "sr-greedy":
            policy = SelectiveReplicationGreedy(verbose=1)
        elif policy_name == "mp-ilp":
            policy = ModelParallelismILP(verbose=1)
        elif policy_name == "mp-search":
            policy = ModelParallelismSearch(verbose=2)
        elif "mp-greedy" in policy_name:
            group_size = int(policy_name.split("-")[2])
            policy = ModelParallelismGreedy(group_size=group_size, verbose=1)
        else:
            raise ValueError(f"Invalid placement policy: {policy_name}")

        placement = policy.place_models(controller, cluster_env, model_datas)
        return placement

    return ServingCase(register_models, generate_workload, place_models)


def simulate_one_all_equal_case(case, prof_database=None):
    serving_case = get_all_equal_serving_case(case, prof_database)
    stats, placement = simulate_one_case(serving_case)
    return stats, placement


def run_one_all_equal_case(case, prof_database=None):
    serving_case = get_all_equal_serving_case(case, prof_database)
    stats, placement = run_one_case(serving_case)
    return stats, placement


_DATA_HEADS = ("exp_name", "num_devices", "mem_budget", "model_type",
               "num_models", "arrival_process", "slo",
               "duration", "policy_name", "placement", "goodput", "mode")


def run_all_equal_cases(cases, exp_name="default", output_file=None,
                        mode="simulate", parallel=False):
    if mode == "simulate":
        if parallel:
            ray.init(address="auto", runtime_env={"working_dir": os.getcwd()},
                     ignore_reinit_error=True)
            run_one_case_ = ray.remote(num_cpus=2)(simulate_one_all_equal_case).remote
        else:
            run_one_case_ = simulate_one_all_equal_case
    else:
        ray.init(address="auto", runtime_env={"working_dir": os.getcwd()},
                 ignore_reinit_error=True)
        run_one_case_ = run_one_all_equal_case

    run_results = []
    for case in cases:
        run_results.append(run_one_case_(case))

    results = []
    for case, run_res in zip(cases, run_results):
        if parallel:
            stats, placement = ray.get(run_res)
        else:
            stats, placement = run_res

        Workload.print_stats(stats)
        goodput = stats.average_goodput

        res = (placement, round(goodput, 3), mode)
        values = (exp_name,) + tuple(case) + res

        if output_file is not None:
            write_tsv(_DATA_HEADS, values, output_file)
        results.append(res)

    return results


def read_all_equal_case_tsv(filename):
    rows = []

    for line in open(filename):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        (exp_name, num_devices, mem_budget, model_type, num_models,
         arrival_process, slo, duration, policy_name, placement,
         goodput, mode) = line.split("\t")

        num_devices = int(num_devices)
        num_models = int(num_models)
        slo = float(slo)
        duration = float(duration)
        goodput = float(goodput)

        values = locals()
        row = {
            key: values[key]
            for key in _DATA_HEADS
        }
        rows.append(row)

    return rows


if __name__ == "__main__":
    policy = "sr-greedy"
    num_devices = 4
    mem_budget = 12 * GB
    model_type = "bert-1.3b"
    num_models = 2
    per_model_rate = 2
    per_model_cv = 4
    slo = 0.4
    duration = 50

    arrival_process = GammaProcess(per_model_rate, per_model_cv)

    cases = [
        AllEqualCase(num_devices, mem_budget, model_type, num_models,
                     arrival_process, slo, duration, policy),]

    run_all_equal_cases(cases,
                        exp_name="tmp",
                        output_file="tmp.tsv",
                        mode="run",
                        parallel=False)

    run_all_equal_cases(cases,
                        exp_name="tmp",
                        output_file="tmp.tsv",
                        mode="simulate",
                        parallel=True)
