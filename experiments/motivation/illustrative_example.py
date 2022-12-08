import numpy as np
import argparse

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_goodput_vs_slo.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--policy", type=str)
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    if args.policy is not None:
        policies = [args.policy]
    else:
        policies = ["sr-greedy", "mp-search"]
    num_devices = 2
    mem_budget = 14 * GB
    model_type = "bert-6.7b"
    num_models = 2
    total_rate = 8
    # choices: {"gamma", "uniform_mmpp"}
    arrival_process = "gamma"
    # choices: {"uniform", "power_law", "triangle_decay"}
    rate_distribution = "uniform"
    arrival_process_kwargs = {"cv": 4}
    slo_scale = np.inf
    duration = 200

    cases = []
    for policy_name in policies:
        cases.append(EqualModelCase(
            num_devices, mem_budget, model_type, num_models,
            total_rate, rate_distribution,
            arrival_process, arrival_process_kwargs,
            slo_scale, duration, policy_name))

    _, stats = run_equal_model_cases(cases,
                                     exp_name=args.exp_name,
                                     output_file=args.output,
                                     mode=args.mode,
                                     parallel=args.parallel)

    print(stats)
