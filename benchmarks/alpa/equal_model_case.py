import argparse
from collections import namedtuple, defaultdict
import os

import numpy as np

import ray

from alpa_serve.simulator.controller import (Controller, simulate_one_case,
    approximate_one_case)
from alpa_serve.simulator.workload import Workload, GammaProcess, UniformMMPP
from alpa_serve.profiling import ProfilingDatabase, ParallelConfig
from alpa_serve.placement_policy import (ClusterEnv, ModelData,
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    SelectiveReplicationSearch,
    ModelParallelismILP, ModelParallelismGreedy, ModelParallelismSearch)
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.trace import Trace, report_group_stats
from alpa_serve.util import GB, write_tsv, ServingCase

from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.run_one_case import run_one_case


# A case where all models are the same
EqualModelCase = namedtuple("EqualModelCase", [
    "num_devices", "mem_budget", "model_type", "num_models",
    "total_rate", "rate_distribution", "arrival_process", "arrival_process_kwargs",
    "slo_scale", "duration", "policy_name"])

# default_slos = {"bert-1.3b": 0.100, "bert-2.6b": 0.145, "bert-6.7b": 0.234,
#                 "moe-1.3b": 0.022, "moe-2.4b": 0.028, "moe-7.1b": 0.041}
# default_slos = {"bert-1.3b": 0.06183671951293945, "bert-2.6b": 0.09547750155131023 , "bert-6.7b": 0.18315919240315756 ,
#                 "moe-1.3b": 0.024369213316175673, "moe-2.4b": 0.028811666700575087 , "moe-7.1b": 0.04116768307156033 }
default_slos = {"bert-1.3b": 0.15109131071302626, "bert-2.6b": 0.23757214016384548, "bert-6.7b": 0.3950637976328532,
                "moe-1.3b": 0.024369213316175673, "moe-2.4b": 0.028811666700575087 , "moe-7.1b": 0.04116768307156033 }


def get_equal_model_serving_case(case, prof_database=None):
    if prof_database is None:
        prof_database = ProfilingDatabase("profiling_result.pkl")

    (num_devices, mem_budget, model_type, num_models,
     total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
     slo_scale, duration, policy_name) = case

    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget)
    num_models = num_models

    model_names = [f"m{i}" for i in range(num_models)]
    model_types = [model_type] * num_models
    # slos = [slo_scale * default_slos[model_type]] * num_models
    single_latency = {
        model_type: sum(prof_database.get(model_type).para_dict[ParallelConfig(1,1,1)
        ].latency[1]) for model_type in set(model_types)}
    slos = [slo_scale * single_latency[model_type]] * num_models

    if rate_distribution == "uniform":
        rates = [total_rate / num_models] * num_models
    elif rate_distribution == "power_law":
        alpha = 0.5
        s = sum((x+1)**(-alpha) for x in range(num_models))
        base = total_rate / s
        rates = [base * ((x+1) ** (-alpha)) for x in range(num_models)]
    elif rate_distribution == "triangle_decay":
        q = 1/2
        frac = [1]
        cur_rate = q
        cnt = int(1 / cur_rate)
        while (len(frac) < num_models):
            for i in range(cnt):
                frac.append(cur_rate)
                if len(frac) == num_models:
                    break
            cur_rate *= q
            cnt = int(1 / cur_rate)
        s = sum(np.array(frac))
        rates = np.array(frac) / s * total_rate
    elif rate_distribution is None:
        pass
    else:
        raise ValueError(f"Invalid rate distribution: {rate_distribution}")

    train_workload = None
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
        train_replays = azure_v2_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time='13.0.0',
                                              end_time='13.23.60',
                                              interval_seconds=5400,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        test_replays = azure_v2_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time='13.0.0',
                                              end_time='13.23.60',
                                              interval_seconds=5400,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        ws = []
        for model_name, slo in zip(model_names, slos):
            ws.append(train_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)

        # for debugging:

        for m in test_replays:
            test_replays[m].report_stats()
        report_group_stats(list(test_replays.values()))
        arrival_processes = [test_replays[model_name] for model_name in model_names]
    elif arrival_process == "azure_v1":
        azure_v1_trace_dir = arrival_process_kwargs["trace_dir"]
        azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
        train_replays = azure_v1_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="exponential",
                                              start_time="0.0.0",
                                              end_time="0.1.0",
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"])
        test_replays = azure_v1_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="exponential",
                                              start_time="0.0.0",
                                              end_time="0.1.0",
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"])
        ws = []
        for model_name, slo in zip(model_names, slos):
            ws.append(train_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)
        # for debugging:
        for m in test_replays:
            test_replays[m].report_stats()
        report_group_stats(list(test_replays.values()))
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
        base_seed = 0
        ws = []
        for i in range(num_models):
            if "azure" in arrival_process:
                ws.append(arrival_processes[i].to_workload(slos[i]))
            else:
                ws.append(arrival_processes[i].generate_workload(
                    model_names[i], start, duration, slo=slos[i], seed=base_seed + i))
        return Workload.merge(*ws)

    def place_models(controller):
        num_models = len(model_names)
        model_datas = []
        for i in range(num_models):
            model_datas.append(ModelData(model_names[i], slos[i], rates[i], cvs[i],
                                         prof_database.get(model_types[i])))

        if policy_name == "sr-ilp":
            policy = SelectiveReplicationILP(verbose=1)
        elif policy_name in "sr-greedy":
            policy = SelectiveReplicationGreedy(verbose=1)
        elif policy_name == "sr-search":
            policy = SelectiveReplicationSearch(verbose=1)
        elif policy_name == "mp-ilp":
            policy = ModelParallelismILP(verbose=1)
        elif policy_name in ["mp-search", "mp-search-evo"]:
            use_evo_search = "evo" in policy_name
            policy = ModelParallelismSearch(
                use_evo_search=use_evo_search, verbose=2)
        elif "mp-greedy" in policy_name:
            group_size = int(policy_name.split("-")[2])
            use_evo_search = "evo" in policy_name
            policy = ModelParallelismGreedy(
                use_evo_search=use_evo_search,
                group_size=group_size, verbose=1)
        else:
            raise ValueError(f"Invalid placement policy: {policy_name}")

        placement = policy.place_models(controller, cluster_env, model_datas, train_workload)

        return placement

    return ServingCase(register_models, generate_workload, place_models)


_DATA_HEADS = ("exp_name",
               "num_devices", "mem_budget", "model_type", "num_models",
               "total_rate", "rate_distribution",
               "arrival_process", "arrival_process_kwargs",
               "slo_scale", "duration", "policy_name",
               "placement", "goodput", "mode")


def run_one_equal_model_case(case, exp_name, mode,
                             output_file=None, prof_database=None,
                             debug=False):
    serving_case = get_equal_model_serving_case(case, prof_database)

    if mode == "simulate":
        stats, placement = approximate_one_case(serving_case, debug=debug)
    else:
        stats, placement = run_one_case(serving_case, debug=debug)

    #Workload.print_stats(stats)
    print(f"group #req: {stats.group_num_requests}")

    res = (placement, round(stats.goodput, 3), mode)
    values = (exp_name,) + tuple(case) + res

    if output_file is not None:
        write_tsv(_DATA_HEADS, values, output_file)

    return values


def run_equal_model_cases(cases, exp_name="default", output_file=None,
                          mode="simulate", debug_tstamp=False, parallel=False):
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env={"working_dir": os.getcwd()})

    if parallel:
        run_one_case_ = ray.remote(num_cpus=2)(run_one_equal_model_case).remote
    else:
        run_one_case_ = run_one_equal_model_case

    results = []
    for case in cases:
        results.append(run_one_case_(case, exp_name, mode,
            output_file=output_file, debug=debug_tstamp))

    if parallel:
        results = ray.get(results)

    return results


def read_equal_model_case_tsv(filename):
    rows = []  # List[dict]

    for line in open(filename):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        (exp_name,
         num_devices, mem_budget, model_type, num_models,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
        help="Whether to do a real run to check the results of simulation.")
    args = parser.parse_args()

    num_devices = 4
    mem_budget = 12 * GB
    model_type = "bert-1.3b"
    num_models = 8
    total_rate = 30
    rate_distribution = "uniform"
    arrival_process = "gamma"
    arrival_process_kwargs = {"cv": 4}
    slo = 0.5
    duration = 50
    policy_name = "mp-greedy-2"

    cases = [
        EqualModelCase(num_devices, mem_budget, model_type, num_models,
                       total_rate, rate_distribution,
                       arrival_process, arrival_process_kwargs,
                       slo, duration, policy_name)
    ]

    if args.run:
        run_equal_model_cases(cases,
                             exp_name="tmp",
                             output_file="tmp.tsv",
                             mode="run",
                             parallel=False)

    run_equal_model_cases(cases,
                          exp_name="tmp",
                          output_file="tmp.tsv",
                          mode="simulate",
                          parallel=True)
