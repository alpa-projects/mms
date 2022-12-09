import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB
from alpa_serve.profiling import ProfilingDatabase

def run_case(case_id=1, mode="simulate", parallel=False):
    policies = ["mp-greedy-8", "sr-uniform"]
    num_devices = 8
    model_type = "bert-2.6b"
    mem_budget = 14 * GB
    num_models = 8
    arrival_process = "gamma"
    rate_distribution = "uniform"
    slo_scale = np.inf
    total_rate = 60
    arrival_process_kwargs = {"cv": 5.0}
    cases = []
    if case_id == 1:
        duration = 500
        total_rates = np.linspace(1, 9.25, 20) * num_devices
        for policy_name in policies:
            for total_rate in total_rates:
                cases.append(EqualModelCase(
                    num_devices, mem_budget, model_type, num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name))
    elif case_id == 2:
        duration = 20000
        cvs = np.linspace(0.1, 8, 20)
        for policy_name in policies:
            for cv in cvs:
                arrival_process_kwargs = {"cv": cv}
                cases.append(EqualModelCase(
                    num_devices, mem_budget, model_type, num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name))
    elif case_id == 3:
        duration = 500
        slo_scales = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for policy_name in policies:
            for slo_scale in slo_scales:
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
    if case_id == 1:
        results = ((policies, total_rates), stats)
    elif case_id == 2:
        results = ((policies, cvs), stats)
    elif case_id == 3:
        results = ((policies, slo_scales), stats)
    with open(f"changing_rate_cv_slo_{case_id}.pkl", "wb") as f:
        pickle.dump(results, f)

def plot_case(case_id=1):
    if case_id == 1:
        with open(f"changing_rate_cv_slo_{case_id}.pkl", "rb") as f:
            ((policies, x), stats) = pickle.load(f)
    elif case_id == 2:
        with open(f"changing_rate_cv_slo_{case_id}.pkl", "rb") as f:
            ((policies, x), stats) = pickle.load(f)
    elif case_id == 3:
        with open(f"changing_rate_cv_slo_{case_id}.pkl", "rb") as f:
            ((policies, x), stats) = pickle.load(f)

    plt.figure()
    i = 0
    for policy in policies:
        y = []
        for _ in x:
            stat = stats[i]
            if case_id in [1, 2]:
                y.append(stat.latency_mean)
            elif case_id == 3:
                y.append(stat.goodput)
            i += 1
        plt.plot(x, y, '.-', label=policy)
    if case_id == 1:
        plt.xlabel("Total Rates (req/s)")
    elif case_id == 2:
        plt.xlabel("Coefficient of Variance")
    elif case_id == 2:
        plt.xlabel("SLO Scale")
    if case_id in [1, 2]:
        plt.ylabel("Mean Latency (s)")
    elif case_id == 3:
        plt.ylabel("Goodput (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"changing_rate_cv_slo_{case_id}.pdf")
    # plt.show()


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
    run_case(case_id=3, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=3)
