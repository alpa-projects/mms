import argparse

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_goodput_vs_slo.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--trace", choices=["synthetic", "azure_v2"],
                        default="synthetic")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4", "mp-search"]
    num_devices = 16
    mem_budget = 14 * GB
    model_type = "bert-2.6b"
    num_models = 24
    total_rate = 70
    if args.trace == "synthetic":
        # choices: {"gamma", "uniform_mmpp"}
        arrival_process = "gamma"
        # choices: {"uniform", "power_law", "triangle_decay"}
        rate_distribution = "power_law"
        arrival_process_kwargs = {"cv": 4}
    elif args.trace == "azure_v2":
        # choices: {"azure_v2"}
        arrival_process = "azure_v2"
        rate_distribution = None
        arrival_process_kwargs = None

    slo_scales = [0.5, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7.5, 10]
    duration = 200

    cases = []
    for slo_scale in slo_scales:
        for policy_name in policies:
            cases.append(EqualModelCase(
                num_devices, mem_budget, model_type, num_models,
                total_rate, rate_distribution,
                arrival_process, arrival_process_kwargs,
                slo_scale, duration, policy_name))

    run_equal_model_cases(cases,
                          exp_name=args.exp_name,
                          output_file=args.output,
                          mode=args.mode,
                          parallel=args.parallel)
