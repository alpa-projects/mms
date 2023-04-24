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
    total_rate = 30
    arrival_process_kwargs = {"cv": 3.0}
    overheads = list(np.linspace(1.0, 1.5, 6)) + [None]
    duration = 500
    results = []
    slo_scales = np.linspace(1.0, 20, 20)
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
                None,
                num_devices, mem_budget, model_type, num_models,
                total_rate, rate_distribution,
                arrival_process, arrival_process_kwargs,
                slo_scale, duration, policy_name,
                None, None, None, None))

        all_results = run_equal_model_cases(cases,
                                        output_file=None,
                                        mode=mode,
                                        parallel=parallel,
                                        prof_database=prof_database,
                                        return_stats_and_placement=True)
        stats = [result[0] for result in all_results]
        results.append((policy_name, overhead, slo_scales, stats))


    with open(f"changing_pipeline_overhead_{case_id}.pkl", "wb") as f:
        pickle.dump(results, f)

def plot_case(case_id=1):
    with open(f"changing_pipeline_overhead_{case_id}.pkl", "rb") as f:
        results = pickle.load(f)

    plt.figure(figsize=(3, 2))
    for policy_name, overhead, slo_scales, stats in results:
        x = slo_scales
        label = policy_name_to_label[policy_name] + ("" if overhead is None else f" ($\\alpha$={overhead})")
        y = []
        for stat in stats:
            y.append(stat.goodput * 100)
        alpha = 1 - (overhead - 1) * 1.6 if overhead is not None else 1
        plt.plot(x, y, '.-', label=label, alpha = alpha, color = color_dict[policy_name])
    plt.xlabel("SLO Scale")
    plt.ylabel("SLO Attainment (%)")
    plt.grid()
    plt.legend(prop={'size': 5.5})
    plt.tight_layout()
    plt.savefig(f"changing_pipeline_overhead_{case_id}.pdf", bbox_inches=Bbox([[0, 0], [3, 2.25]]))
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()
    run_case(case_id=1, mode=args.mode, parallel=args.parallel)
    plot_case(case_id=1)
