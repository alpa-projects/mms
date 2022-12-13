import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB

color_dict = {
    "sr-greedy": "C1",
    "mp-search": "C0",
}
policy_name_to_label = {
    "sr-greedy": "Replication Only",
    "mp-search": "Model Parallelism",
}


def run_case(rate_distribution=(1, 1), cv=1.0, case_id=1, mode="simulate", parallel=False):
    policies = ["sr-greedy", "mp-search"]
    num_devices = 2
    mem_budget = 14 * GB
    model_type = "bert-6.7b"
    num_models = 2
    total_rate = 3
    arrival_process = "gamma"
    arrival_process_kwargs = {"cv": cv}
    slo_scale = np.inf
    duration = 20000

    cases = []
    for policy_name in policies:
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
    policies_and_stats = (policies, stats)
    with open(f"illustrative_example_policies_and_stats_{case_id}.pkl", "wb") as f:
        pickle.dump(policies_and_stats, f)

def plot_case(case_id=1, per_model_curves=False):
    with open(f"illustrative_example_policies_and_stats_{case_id}.pkl", "rb") as f:
        policies, stats = pickle.load(f)
    plt.figure(figsize=(4.5, 3.5))
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
    plt.xlabel("Processing Latency (s)")
    plt.ylabel("CDF")
    plt.xlim(0, max_latency)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"illustrative_example_{case_id}.pdf")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()
    # run_case((1, 1), cv=1, case_id=1, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=1)
    # run_case((2, 8), cv=1, case_id=2, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=2, per_model_curves=True)
    # run_case((1, 1), cv=3.0, case_id=3, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=3)
