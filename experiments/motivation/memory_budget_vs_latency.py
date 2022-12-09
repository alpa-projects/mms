import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB
from alpa_serve.profiling import ProfilingDatabase

def run_case(case_id=1, mode="simulate", parallel=False):
    prof_database = ProfilingDatabase("profiling_result.pkl")
    num_devices = 8
    model_type = "bert-1.3b"
    weight_mem = prof_database.get(model_type).para_dict[(1,1,1)].weight_mem[0]
    num_models = 8
    total_rate = 60
    arrival_process = "gamma"
    rate_distribution = "uniform"
    arrival_process_kwargs = {"cv": 5.0}
    slo_scale = np.inf
    duration = 20000
    mp_mem_budgets = [weight_mem, weight_mem * 2, weight_mem * 4]
    mp_policies = ["mp-greedy-8", "mp-greedy-4", "mp-greedy-2"]
    sr_mem_budgets = [weight_mem * i for i in range(1, 6)]
    sr_policies = ["sr-greedy"] * len(sr_mem_budgets)

    cases = []
    for policy_name, mem_budget in zip(mp_policies, mp_mem_budgets):
        cases.append(EqualModelCase(
            num_devices, mem_budget, model_type, num_models,
            total_rate, rate_distribution,
            arrival_process, arrival_process_kwargs,
            slo_scale, duration, policy_name))

    for policy_name, mem_budget in zip(sr_policies, sr_mem_budgets):
        cases.append(EqualModelCase(
            num_devices, mem_budget, model_type, num_models,
            total_rate, rate_distribution,
            arrival_process, arrival_process_kwargs,
            slo_scale, duration, policy_name))


    _, stats = run_equal_model_cases(cases,
                                     exp_name=None,
                                     output_file=None,
                                     mode=mode,
                                     parallel=parallel)
    results = ((mp_mem_budgets, mp_policies, sr_mem_budgets, sr_policies), stats)
    with open(f"memory_budget_vs_latency_results_{case_id}.pkl", "wb") as f:
        pickle.dump(results, f)

def plot_case(case_id=1):
    with open(f"memory_budget_vs_latency_results_{case_id}.pkl", "rb") as f:
        ((mp_mem_budgets, mp_policies, sr_mem_budgets, sr_policies), stats) = pickle.load(f)

    mp_x = mp_mem_budgets
    mp_y = []
    mp_stats = stats[:len(mp_policies)]
    for stat in mp_stats:
        mp_y.append(stat.latency_mean)

    sr_x = sr_mem_budgets
    sr_y = []
    sr_stats = stats[:len(sr_policies)]
    for stat in sr_stats:
        sr_y.append(stat.latency_mean)


    plt.figure()
    plt.plot(mp_x, mp_y, label="MP")
    plt.plot(sr_x, sr_y, label="SR")
    plt.xlabel("Memory Budget (Bytes)")
    plt.ylabel("Mean Latency (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"memory_budget_vs_latency_{case_id}.pdf")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()
    run_case(case_id=1, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=1)
