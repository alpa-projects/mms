import argparse

from benchmarks.alpa.all_equal_case import AllEqualCase, run_all_equal_cases
from alpa_serve.util import GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_goodput_vs_slo.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4", "mp-greedy-8"]
    num_devices = 8
    mem_budget = 10 * GB
    model_type = "bert-1.3b"
    num_models = 16
    per_model_rate = 4
    per_model_cv = 4
    slos = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 8.0]
    duration = 200

    cases = []
    for slo in slos:
        for policy in policies:
            cases.append(AllEqualCase(
                num_devices, mem_budget, model_type, num_models,
                per_model_rate, per_model_cv, slo, duration, policy))

    run_all_equal_cases(cases,
                        exp_name=args.exp_name,
                        output_file=args.output,
                        mode=args.mode,
                        parallel=args.parallel)
