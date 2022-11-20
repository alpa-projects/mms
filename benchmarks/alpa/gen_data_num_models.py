import argparse

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_num_models.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4", "mp-search"]
    num_devices_list = [4, 8, 12, 16]
    mem_budget = 12 * GB
    model_type = "bert-2.6b"
    num_models_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    per_model_rate = 2
    rate_distribution = "uniform"
    arrival_process = "gamma"
    arrival_process_kwargs = {"cv": 4}
    slos = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
    duration = 200

    cases = []
    for num_devices in num_devices_list:
        for slo in slos:
            for policy_name in policies:
                for num_models in num_models_list:
                    total_rate = per_model_rate * num_models
                    cases.append(EqualModelCase(
                        num_devices, mem_budget, model_type, num_models,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        slo, duration, policy_name))

    run_equal_model_cases(cases,
                          exp_name=args.exp_name,
                          output_file=args.output,
                          mode=args.mode,
                          parallel=args.parallel)
