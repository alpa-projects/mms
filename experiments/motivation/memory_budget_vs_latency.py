import pickle
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Pdf')
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB
from alpa_serve.profiling import ProfilingDatabase

color_dict = {
    "sr-greedy": "C1",
    "mp-search": "C0",
}
policy_name_to_label = {
    "sr": "Replication",
    "mp": "Model Parallelism",
}


def run_case(case_id=1, mode="simulate", parallel=False):
    prof_database = ProfilingDatabase("profiling_result.pkl")
    num_devices = 8
    num_models = 8
    arrival_process = "gamma"
    rate_distribution = "uniform"
    arrival_process_kwargs = {"cv": 3.0}
    slo_scale = np.inf
    duration = 20000
    if case_id == 1:
        total_rate = 30
        model_type = "bert-1.3b"
        mp_mem_budgets = [3.5 * GB, 6 * GB, 11 * GB, 21.6 * GB]
        mp_policies = ["mp-greedy-8", "mp-greedy-4", "mp-greedy-2", "sr-uniform"]
        sr_mem_budgets = [2.7 * GB * i for i in range(1, 9)]
        sr_policies = ["sr-uniform"] * len(sr_mem_budgets)
    elif case_id == 2:
        total_rate = 20
        model_type = "bert-2.6b"
        mp_mem_budgets = [7 * GB, 12 * GB, 22 * GB, 44 * GB]
        mp_policies = ["mp-greedy-8", "mp-greedy-4", "mp-greedy-2", "sr-uniform"]
        sr_mem_budgets = [5.5 * GB * i for i in range(1, 9)]
        sr_policies = ["sr-uniform"] * len(sr_mem_budgets)

    cases = []
    for policy_name, mem_budget in zip(mp_policies, mp_mem_budgets):
        cases.append(EqualModelCase(
            None,
            num_devices, mem_budget, model_type, num_models,
            total_rate, rate_distribution,
            arrival_process, arrival_process_kwargs,
            slo_scale, duration, policy_name,
            None, None, None, None))

    for policy_name, mem_budget in zip(sr_policies, sr_mem_budgets):
        cases.append(EqualModelCase(
            None,
            num_devices, mem_budget, model_type, num_models,
            total_rate, rate_distribution,
            arrival_process, arrival_process_kwargs,
            slo_scale, duration, policy_name,
            None, None, None, None))


    results = run_equal_model_cases(cases,
                                    output_file=None,
                                    mode=mode,
                                    parallel=parallel,
                                    return_stats_and_placement=True)

    stats = [result[0] for result in results]
    results = ((mp_mem_budgets, mp_policies, sr_mem_budgets, sr_policies), stats)
    with open(f"memory_budget_vs_latency_results_{case_id}.pkl", "wb") as f:
        pickle.dump(results, f)

def get_latency_percentile(stat, p=99):
    all_latencies = []
    for per_model_stat in stat.per_model_stats:
        all_latencies.extend(per_model_stat.latency)
    return np.percentile(all_latencies, p)

def plot_case(case_id=1):
    with open(f"memory_budget_vs_latency_results_{case_id}.pkl", "rb") as f:
        ((mp_mem_budgets, mp_policies, sr_mem_budgets, sr_policies), stats) = pickle.load(f)
    mp_x = np.array(mp_mem_budgets) / GB
    mp_y = []
    mp_y_p99 = []
    mp_stats = stats[:len(mp_policies)]
    for stat in mp_stats:
        mp_y.append(stat.latency_mean)
        mp_y_p99.append(get_latency_percentile(stat, 99))

    sr_x = np.array(sr_mem_budgets) / GB
    sr_y = []
    sr_stats = stats[len(mp_policies):]
    sr_y_p99 = []
    for stat in sr_stats:
        sr_y.append(stat.latency_mean)
        sr_y_p99.append(get_latency_percentile(stat, 99))

    plt.figure(figsize=(3, 2))
    plt.plot(mp_x, mp_y, '.-', label=policy_name_to_label["mp"])
    plt.plot(sr_x, sr_y, '.-', label=policy_name_to_label["sr"])
    plt.axvline(13, linestyle='--', color = "black", label = "GPU Memory Bound", linewidth=0.75)
    plt.xlabel("Memory Budget (GB)")
    plt.ylabel("Mean Latency (s)")
    plt.grid()
    plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig(f"memory_budget_vs_latency_mean_latency_{case_id}.pdf")

    plt.figure(figsize=(3, 2))
    plt.plot(mp_x, mp_y_p99, '.-', label=policy_name_to_label["mp"])
    plt.plot(sr_x, sr_y_p99, '.-', label=policy_name_to_label["sr"])
    plt.axvline(13, linestyle='--', color = "black", label = "GPU Memory Bound", linewidth=0.75)
    plt.xlabel("Memory Budget (GB)")
    plt.ylabel("P99 Latency (s)")
    plt.grid()
    plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig(f"memory_budget_vs_latency_p99_latency_{case_id}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()
    run_case(case_id=1, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=1)
    run_case(case_id=2, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=2)
