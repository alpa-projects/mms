import pickle
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Pdf')
import matplotlib.pyplot as plt
import functools

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB

color_dict = {
    "sr-greedy": "C1",
    "mp-search": "C0",
}
policy_name_to_label = {
    "sr-greedy": "Simple Placement",
    "mp-search": "Model Parallelism",
}


def run_case(rate_distribution=(1, 1), cv=1.0, case_id=1, mode="simulate", parallel=False, duration=20000):
    policies = ["sr-greedy", "mp-search"]
    num_devices = 2
    mem_budget = 14 * GB
    model_type = "bert-6.7b"
    num_models = 2
    total_rate = 3
    arrival_process = "gamma"
    arrival_process_kwargs = {"cv": cv}
    slo_scale = np.inf

    cases = []
    for policy_name in policies:
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
    policies_and_stats = (policies, stats)
    with open(f"illustrative_example_policies_and_stats_{case_id}.pkl", "wb") as f:
        pickle.dump(policies_and_stats, f)

def plot_case(case_id=1, per_model_curves=False, figsize=(4.5, 3.5), legend_size=8):
    with open(f"illustrative_example_policies_and_stats_{case_id}.pkl", "rb") as f:
        policies, stats = pickle.load(f)
    plt.figure(figsize=figsize)
    max_latency = 0
    for policy_name, stat in zip(policies, stats):
        for model_id, model_stat in enumerate(stat.per_model_stats):
            latency = model_stat.latency
            max_latency = max(max_latency, max(latency))
    for policy_name, stat in zip(policies, stats):
        all_latency = []
        for model_id, model_stat in enumerate(stat.per_model_stats):
            latency = model_stat.latency
            all_latency.extend(latency)
        mean_latency = np.mean(all_latency)
        all_latency.extend([max_latency, 0])
        cdfx = np.sort(all_latency)
        cdfy = np.linspace(0, 1, len(all_latency), endpoint=False)
        plt.plot(cdfx, cdfy, label=policy_name_to_label[policy_name], color=color_dict[policy_name])
        if per_model_curves:
            for model_id, model_stat in enumerate(stat.per_model_stats):
                latency = model_stat.latency
                cdfx = np.sort(latency)
                cdfy = np.linspace(0, 1, len(latency), endpoint=False)
                plt.plot(cdfx, cdfy, label=policy_name_to_label[policy_name] + f" Model {model_id + 1}", color=color_dict[policy_name], alpha=(len(stat.per_model_stats) - model_id) * 0.3 + 0.15)
        plt.axvline(mean_latency, linestyle='--', color = color_dict[policy_name], label = policy_name_to_label[policy_name]+' Mean Latency', linewidth=0.75)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.grid()
    plt.xlim(0, max_latency)
    plt.ylim(0, 1.05)
    plt.legend(prop={'size': legend_size})
    plt.tight_layout()
    plt.savefig(f"illustrative_example_{case_id}.pdf")
    # plt.show()

def cmp(t1, t2):
    if np.abs(t1[0] - t2[0]) > 1e-4 or t1[1] == t2[1]:
        return np.sign(t1[0] - t2[0])
    else:
        return np.sign(t1[1] - t2[1])

def plot_case_utilization(case_id=1, per_model_curves=False):
    with open(f"illustrative_example_policies_and_stats_{case_id}.pkl", "rb") as f:
        policies, stats = pickle.load(f)
    max_latency = 0
    plt.figure(figsize=(3.5, 2.5))
    for policy_name, stat in zip(policies, stats):
        all_time_stamps = []
        for model_id, model_stat in enumerate(stat.per_model_stats):
            starts = [(t - 0.39624744, 1) for t in model_stat.request_finishes]
            finishes = [(t, -1) for t in model_stat.request_finishes]
            all_time_stamps.extend(starts)
            all_time_stamps.extend(finishes)
        all_time_stamps = sorted(all_time_stamps, key=functools.cmp_to_key(cmp))
        x = np.linspace(0, 40, 400)
        y = []
        i = 0
        u = 0
        for t in x:
            while all_time_stamps[i][0] <= t and i < len(all_time_stamps):
                u += all_time_stamps[i][1]
                i += 1
            y.append(u/2 * 100)
        cut_i0 = 7 * 10
        cut_i1 = 12 * 10
        cut_i2 = 17 * 10
        cut_i3 = 37 * 10
        x = np.concatenate((x[cut_i0:cut_i1] - x[cut_i0], x[cut_i2:cut_i3] - (x[cut_i2] - x[cut_i0])))
        y = np.concatenate((y[cut_i0:cut_i1], y[cut_i2:cut_i3]))

        plt.plot(x, y, label=policy_name_to_label[policy_name], color=color_dict[policy_name])
        plt.fill_between(x, y, alpha=0.2, color=color_dict[policy_name])
    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.legend(prop={'size': 8}, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
    plt.tight_layout()
    plt.savefig(f"illustrative_example_utilization_{case_id}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()
    run_case((1, 1), cv=1, case_id=1, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=1, figsize=(3.5, 2.5))
    run_case((2, 8), cv=1, case_id=2, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=2, figsize=(3.5, 2.5), per_model_curves=True, legend_size=6)
    run_case((1, 1), cv=3.0, case_id=3, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=3, figsize=(3.5, 2.5))
    run_case((1, 1), cv=3.0, case_id=4, mode=args.mode, parallel=args.parallel, duration=500)
    plot_case_utilization(case_id=4)
