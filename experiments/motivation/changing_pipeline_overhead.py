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
    overheads = list(np.linspace(1.0, 1.5, 11)) + [None]
    duration = 500
    results = []
    slo_scales = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for overhead in overheads:
        if overhead is None:
            prof_database = None
            policy_name = "sr-uniform"
        else:
            prof_database = ProfilingDatabase("profiling_result.pkl")
            single_device_latency = (prof_database.results[model_type]
                                     .para_dict[(1,1,1)].latency[1][0])
            (prof_database.results[model_type]
             .para_dict[(1, 1, num_devices)].latency[1]) = [
                overhead * single_device_latency / num_devices] * num_devices
            policy_name = "mp-greedy-8"
        cases = []
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
                                         parallel=parallel,
                                         prof_database=prof_database)
        results.append((policy_name, overhead, slo_scales, stats))


    with open(f"changing_pipeline_overhead_{case_id}.pkl", "wb") as f:
        pickle.dump(results, f)

def plot_case(case_id=1):
    with open(f"changing_pipeline_overhead_{case_id}.pkl", "rb") as f:
        results = pickle.load(f)

    plt.figure()
    for policy_name, overhead, slo_scales, stats in results:
        x = slo_scales
        label = policy_name + ("" if overhead is None else f" (overhead={overhead})")
        y = []
        for stat in stats:
            y.append(stat.goodput)
        plt.plot(x, y, '.-', label=label)
    plt.xlabel("SLO Scale")
    plt.ylabel("Goodput (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"changing_pipeline_overhead_{case_id}.pdf")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()
    run_case(case_id=1, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=1)
