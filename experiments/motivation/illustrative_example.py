import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB

def run_case(rate_distribution=(1, 1), cv=1.0, case_id=1, mode="simulate", parallel=False):
    policies = ["sr-greedy", "mp-search"]
    num_devices = 2
    mem_budget = 14 * GB
    model_type = "bert-6.7b"
    num_models = 2
    total_rate = 6
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

def plot_case(case_id=1):
    with open(f"illustrative_example_policies_and_stats_{case_id}.pkl", "rb") as f:
        policies, stats = pickle.load(f)
    plt.figure()
    for policy_name, stat in zip(policies, stats):
        # import ipdb; ipdb.set_trace()
        print(f"Policy: {policy_name}")
        all_latency = []
        for model_id, model_stat in enumerate(stat.per_model_stats):
            latency = model_stat.latency
            all_latency.extend(latency)
            cdfx = np.sort(latency)
            cdfy = np.linspace(0, 1, len(latency), endpoint=False)
            plt.plot(cdfx, cdfy, label=policy_name + f" model {model_id}")
        cdfx = np.sort(all_latency)
        cdfy = np.linspace(0, 1, len(all_latency), endpoint=False)
        plt.plot(cdfx, cdfy, label=policy_name + f" all")
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
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
    run_case((1, 1), cv=1, case_id=1, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=1)
    run_case((2, 8), cv=1, case_id=2, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=2)
    run_case((2, 8), cv=5, case_id=3, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=3)
