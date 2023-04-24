import pickle
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Pdf')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB
from alpa_serve.profiling import ProfilingDatabase

color_dict = {
    "sr-uniform": "C1",
    "mp-greedy-8": "C0",
}
policy_name_to_label = {
    "sr-uniform": "Replication",
    "mp-greedy-8": "Model Parallelism",
}

def run_case(case_id=1, mode="simulate", parallel=False):
    policies = ["mp-greedy-8", "sr-uniform"]
    num_devices = 8
    model_type = "bert-2.6b"
    mem_budget = 13 * GB
    num_models = 8
    arrival_process = "gamma"
    rate_distribution = "uniform"
    slo_scale = np.inf
    total_rate = 20
    arrival_process_kwargs = {"cv": 3.0}
    cases = []
    if case_id == 1:
        duration = 1000
        total_rates = np.linspace(0.1, 3.0, 20) * num_devices
        for policy_name in policies:
            for total_rate in total_rates:
                cases.append(EqualModelCase(
                    None,
                    num_devices, mem_budget, model_type, num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name,
                    None, None, None, None))
    elif case_id == 2:
        duration = 100000
        cvs = np.linspace(0.1, 8, 20)
        for policy_name in policies:
            for cv in cvs:
                arrival_process_kwargs = {"cv": cv}
                cases.append(EqualModelCase(
                    None,
                    num_devices, mem_budget, model_type, num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name,
                    None, None, None, None))
    elif case_id == 3:
        duration = 500
        total_rate = 30
        slo_scales = np.linspace(1.0, 20, 20)
        for policy_name in policies:
            for slo_scale in slo_scales:
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
    if case_id == 1:
        results = ((policies, total_rates), stats)
    elif case_id == 2:
        results = ((policies, cvs), stats)
    elif case_id == 3:
        results = ((policies, slo_scales), stats)
    with open(f"changing_rate_cv_slo_{case_id}.pkl", "wb") as f:
        pickle.dump(results, f)

def get_latency_percentile(stat, p=99):
    all_latencies = []
    for per_model_stat in stat.per_model_stats:
        all_latencies.extend(per_model_stat.latency)
    return np.percentile(all_latencies, p)

def plot_case(case_id=1):
    if case_id == 1 or case_id == 1.5:
        with open(f"changing_rate_cv_slo_{int(case_id)}.pkl", "rb") as f:
            ((policies, x), stats) = pickle.load(f)
    elif case_id == 2 or case_id == 2.5:
        with open(f"changing_rate_cv_slo_{int(case_id)}.pkl", "rb") as f:
            ((policies, x), stats) = pickle.load(f)
    elif case_id == 3:
        with open(f"changing_rate_cv_slo_{case_id}.pkl", "rb") as f:
            ((policies, x), stats) = pickle.load(f)

    plt.figure(figsize=(3, 2))
    i = 0
    for policy in policies:
        y = []
        for _ in x:
            stat = stats[i]
            if case_id in [1, 2]:
                y.append(stat.latency_mean)
            elif case_id in [1.5, 2.5]:
                y.append(get_latency_percentile(stat))
            elif case_id == 3:
                y.append(stat.goodput * 100)
            i += 1
        plt.plot(x, y, '.-', label=policy_name_to_label[policy],
                 color=color_dict[policy])
    if case_id == 1 or case_id == 1.5:
        plt.xlabel("Total Rates (req/s)")
    elif case_id == 2 or case_id == 2.5:
        plt.xlabel("Coefficient of Variance")
    elif case_id == 3:
        plt.xlabel("SLO Scale")
    if case_id in [1, 2]:
        plt.ylabel("Mean Latency (s)")
    elif case_id in [1.5, 2.5]:
        plt.ylabel("P99 Latency (s)")
    elif case_id == 3:
        # plt.ylim(0, 100)
        plt.ylabel("SLO Attainment (%)")
    plt.grid()
    plt.legend(prop={'size': 8})
    plt.tight_layout()
    if case_id == 3:
        plt.savefig(f"changing_rate_cv_slo_{case_id}.pdf", bbox_inches=Bbox([[0, 0], [3, 2.25]]))
    else:
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
    plot_case(case_id=1.5)
    run_case(case_id=2, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=2)
    plot_case(case_id=2.5)
    run_case(case_id=3, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=3)
