import argparse

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_various_metrics.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4"]
    mem_budget = 12 * GB
    model_type = "bert-2.6b"
    rate_distribution = "power_law"
    arrival_process = "gamma"
    duration = 200

    fixed_num_devices = 8
    fixed_num_models = 8
    fixed_per_model_rate = 3
    fixed_total_rate = fixed_num_models * fixed_per_model_rate
    fixed_slo = 0.8
    fixed_cv = {"cv": 4}

    num_devices_list = [4, 8, 12, 16, 20, 24, 28, 32, 36]
    num_models_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    total_rates = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    slos = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    cvs = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    ##### goodput vs num_devices #####
    cases = []
    for num_devices in num_devices_list:
        for policy_name in policies:
            cases.append(EqualModelCase(
                num_devices, mem_budget, model_type, fixed_num_models,
                fixed_total_rate, rate_distribution,
                arrival_process, fixed_cv,
                fixed_slo, duration, policy_name))

    run_equal_model_cases(cases, exp_name="goodput_vs_num_devices",
                          output_file=args.output,
                          mode=args.mode, parallel=args.parallel)

    ##### goodput vs num_models #####
    cases = []
    for num_models in num_models_list:
        for policy_name in policies:
            cases.append(EqualModelCase(
                fixed_num_devices, mem_budget, model_type, num_models,
                fixed_per_model_rate * num_models, rate_distribution,
                arrival_process, fixed_cv,
                fixed_slo, duration, policy_name))

    run_equal_model_cases(cases, exp_name="goodput_vs_num_models",
                          output_file=args.output,
                          mode=args.mode, parallel=args.parallel)

    ##### goodput vs total_rate #####
    cases = []
    for total_rate in total_rates:
        for policy_name in policies:
            cases.append(EqualModelCase(
                fixed_num_devices, mem_budget, model_type, fixed_num_models,
                total_rate, rate_distribution,
                arrival_process, fixed_cv,
                fixed_slo, duration, policy_name))

    run_equal_model_cases(cases, exp_name="goodput_vs_total_rate",
                          output_file=args.output,
                          mode=args.mode, parallel=args.parallel)

    ##### goodput vs slo #####
    cases = []
    for slo in slos:
        for policy_name in policies:
            cases.append(EqualModelCase(
                fixed_num_devices, mem_budget, model_type, fixed_num_models,
                fixed_total_rate, rate_distribution,
                arrival_process, fixed_cv,
                slo, duration, policy_name))

    run_equal_model_cases(cases, exp_name="goodput_vs_slo",
                          output_file=args.output,
                          mode=args.mode, parallel=args.parallel)

    ##### goodput vs cv #####
    cases = []
    for cv in cvs:
        for policy_name in policies:
            cases.append(EqualModelCase(
                fixed_num_devices, mem_budget, model_type, fixed_num_models,
                fixed_total_rate, rate_distribution,
                arrival_process, {"cv": cv},
                fixed_slo, duration, policy_name))

    run_equal_model_cases(cases, exp_name="goodput_vs_cv",
                          output_file=args.output,
                          mode=args.mode, parallel=args.parallel)
