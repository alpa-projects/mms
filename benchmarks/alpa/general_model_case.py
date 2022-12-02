from collections import namedtuple
import os

import ray

from alpa_serve.simulator.controller import Controller, simulate_one_case
from alpa_serve.simulator.workload import Workload, GammaProcess, UniformMMPP
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.placement_policy import (ClusterEnv, ModelData,
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    ModelParallelismILP, ModelParallelismGreedy, ModelParallelismSearch)
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.trace import Trace
from alpa_serve.util import GB, write_tsv, ServingCase

from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.run_one_case import run_one_case


GeneralModelCase = namedtuple("GeneralModelCase", [
    "num_devices", "mem_budget", "model_types", "model_names",
    "total_rate", "rate_distribution", "arrival_process", "arrival_process_kwargs",
    "slo_scale", "duration", "policy_name"])

default_slos = {"bert-1.3b": 0.5, "bert-2.6b": 0.8, "bert-6.7b": 1.2,
                "moe-1.3b": 0.1, "moe-2.4b": 0.15, "moe-7.1b": 0.2}

def get_general_model_serving_case(case, prof_database=None):
    assert isinstance(case, GeneralModelCase), "not GeneralModelCase"     
    if prof_database is None:
        prof_database = ProfilingDatabase("profiling_result.pkl")

    (num_devices, mem_budget, model_types, model_names,
     total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
     slo_scale, duration, policy_name) = case

    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget)
    assert len(model_names) == len(model_types)
    num_models = len(model_names)

    slos = [default_slos[model_type] * slo_scale for model_type in model_types]

    if rate_distribution == "uniform":
        rates = [total_rate / num_models] * num_models
    elif rate_distribution == "power_law":
        q = 1/2
        s = (1 - q ** num_models) / (1 - q)
        base = total_rate / s
        rates = [base * (q ** i) for i in range(num_models)]
    elif rate_distribution is None:
        pass
    else:
        raise ValueError(f"Invalid rate distribution: {rate_distribution}")

    if arrival_process == "gamma":
        arrival_processes = [
            GammaProcess(rates[i], arrival_process_kwargs["cv"])
            for i in range(num_models)
        ]
    elif arrival_process == "uniform_mmpp":
        arrival_processes = [
            UniformMMPP(**arrival_process_kwargs)
            for _ in range(num_models)
        ]
    elif arrival_process == "azure_v2":
        azure_v2_trace_dir = arrival_process_kwargs["trace_dir"]
        azure_v2_trace = Trace("azure_v2", azure_v2_trace_dir)
        train_replays = azure_v2_trace.replay(model_names, model_mapping_strategy="stripe", arrival_distribution="vanilla",
                                                    start_time='0.0.0', end_time='1.0.0', replication_factor=arrival_process_kwargs["rate_scale"])
        test_replays = azure_v2_trace.replay(model_names,
                                             model_mapping_strategy="stripe",
                                             arrival_distribution="gamma",
                                             start_time='5.0.0',
                                             end_time='6.0.0',
                                             interval_seconds=5400,
                                             rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                             cv_scale_factor=arrival_process_kwargs["cv_scale"])
        train_workload = Workload.empty()
        for model_name, slo in zip(model_names, slos):
            train_workload += train_replays[model_name].to_workload(slo)
        arrival_processes = [test_replays[model_name] for model_name in model_names]
    else:
        raise ValueError("Invalid arrival process: {arrival_process}")

    rates = [a.rate() for a in arrival_processes]
    cvs = [a.cv() for a in arrival_processes]

    def register_models(controller):
        is_simulator = isinstance(controller, Controller)

        for model_name, model_type in zip(model_names, model_types):
            controller.register_model.remote(
                model_name, get_model_def(model_type, is_simulator,
                                          prof_database))

    def generate_workload(start=0):
        w = Workload.empty()
        for i in range(num_models):
            if "azure" in arrival_process:
                w += arrival_processes[i].to_workload(slos[i])
            else:
                w += arrival_processes[i].generate_workload(model_names[i], start,
                                                            duration, slo=slos[i], seed=i)
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

        if "azure" in arrival_process:
            placement = policy.place_models(controller, cluster_env, model_datas, train_workload)
        else:
            placement = policy.place_models(controller, cluster_env, model_datas)

        return placement

    return ServingCase(register_models, generate_workload, place_models)


def simulate_one_general_model_case(case, prof_database=None):
    serving_case = get_general_model_serving_case(case, prof_database)
    stats, placement = simulate_one_case(serving_case)
    return stats, placement


def run_one_general_model_case(case, prof_database=None):
    serving_case = get_general_model_serving_case(case, prof_database)
    stats, placement = run_one_case(serving_case)
    return stats, placement


_DATA_HEADS = ("exp_name", "num_models",
               "num_devices", "mem_budget", "total_rate", "rate_distribution",
               "arrival_process", "arrival_process_kwargs", "slo_scale", "duration",
               "policy_name", "placement", "goodput", "mode")

def run_general_model_cases(cases, exp_name="default", output_file=None,
                            mode="simulate", parallel=False):
    if mode == "simulate":
        if parallel:
            ray.init(address="auto", runtime_env={"working_dir": os.getcwd()},
                     ignore_reinit_error=True)
            run_one_case_ = ray.remote(num_cpus=2)(simulate_one_general_model_case).remote
        else:
            run_one_case_ = simulate_one_general_model_case
    else:
        ray.init(address="auto", runtime_env={"working_dir": os.getcwd()},
                 ignore_reinit_error=True)
        run_one_case_ = run_one_general_model_case

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
        goodput = stats.goodput

        (num_devices, mem_budget, model_types, model_names,
        total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
        slo_scale, duration, policy_name) = case

        case_info = (num_devices, mem_budget, total_rate,
                     rate_distribution, arrival_process,
                     arrival_process_kwargs, slo_scale,
                     duration, policy_name)
        res = (placement, round(goodput, 3), mode)
        values = (exp_name, len(model_types)) + case_info + res

        if output_file is not None:
            write_tsv(_DATA_HEADS, values, output_file)
        results.append(res)

    return results


def read_general_model_case_tsv(filename):
    rows = []  # List[dict]

    for line in open(filename):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        (exp_name, num_models,
         num_devices, mem_budget,
         total_rate, rate_distribution,
         arrival_process, arrival_process_kwargs,
         slo_scale, duration, policy_name,
         placement, goodput, mode) = line.split("\t")

        num_devices = int(num_devices)
        num_models = int(num_models)
        total_rate = float(total_rate) 
        arrival_process_kwargs = eval(arrival_process_kwargs)
        slo_scale = float(slo_scale)
        duration = float(duration)
        goodput = float(goodput)

        values = locals()
        row = {
            key: values[key]
            for key in _DATA_HEADS
        }
        rows.append(row)

    return rows
