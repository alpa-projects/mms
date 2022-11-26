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
    mem_budget = 16 * GB
    model_type = "bert-2.6b"

    # real trace does not need these config
    rate_distribution = None
    total_rate = -1
    duration = -1

    # default configuration
    fixed_num_devices = 64
    fixed_num_models = 64
    fixed_rate_scale = 1
    fixed_cv_scale = 1
    fixed_slo = 0.8

    # real trace related
    arrival_process = "azure_v2"
    arrival_process_kwargs = {"rate_scale": fixed_rate_scale, 
                              "cv_scale": fixed_cv_scale}

    num_devices_list = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    num_models_list = [32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
    rate_scales = [1, 2, 3, 4, 5, 6, 7, 8]
    cv_scales = [1, 2, 3, 4, 5, 6, 7, 8]
    slos = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]

    ##### goodput vs num_devices #####
    # cases = []
    # for num_devices in num_devices_list:
    #     for policy_name in policies:
    #         cases.append(EqualModelCase(
    #             num_devices, mem_budget, model_type, fixed_num_models,
    #             total_rate, rate_distribution,
    #             arrival_process, arrival_process_kwargs,
    #             fixed_slo, duration, policy_name))

    # run_equal_model_cases(cases, exp_name="goodput_vs_num_devices",
    #                       output_file=args.output,
    #                       mode=args.mode, parallel=args.parallel)

    # ##### goodput vs num_models #####
    # cases = []
    # for num_models in num_models_list:
    #     for policy_name in policies:
    #         cases.append(EqualModelCase(
    #             fixed_num_devices, mem_budget, model_type, num_models,
    #             total_rate, rate_distribution,
    #             arrival_process, arrival_process_kwargs,
    #             fixed_slo, duration, policy_name))

    # run_equal_model_cases(cases, exp_name="goodput_vs_num_models",
    #                       output_file=args.output,
    #                       mode=args.mode, parallel=args.parallel)

    # ##### goodput vs slo #####
    # cases = []
    # for slo in slos:
    #     for policy_name in policies:
    #         cases.append(EqualModelCase(
    #             fixed_num_devices, mem_budget, model_type, fixed_num_models,
    #             total_rate, rate_distribution,
    #             arrival_process, arrival_process_kwargs,
    #             slo, duration, policy_name))

    # run_equal_model_cases(cases, exp_name="goodput_vs_slo",
    #                       output_file=args.output,
    #                       mode=args.mode, parallel=args.parallel)

    # ##### goodput vs rate_scale #####
    cases = []
    for rate_scale in rate_scales:
        for policy_name in policies:
            arrival_process_kwargs = {"rate_scale": rate_scale, 
                                      "cv_scale": fixed_cv_scale}
            cases.append(EqualModelCase(
                fixed_num_devices, mem_budget, model_type, fixed_num_models,
                total_rate, rate_distribution,
                arrival_process, arrival_process_kwargs,
                fixed_slo, duration, policy_name))

    run_equal_model_cases(cases, exp_name="goodput_vs_rate_scale",
                          output_file=args.output,
                          mode=args.mode, parallel=args.parallel)

    # ##### goodput vs cv_scale #####
    # cases = []
    # for cv_scale in cv_scales:
    #     for policy_name in policies:
    #         arrival_process_kwargs = {"rate_scale": fixed_rate_scale, 
    #                                   "cv_scale": cv_scale}
    #         cases.append(EqualModelCase(
    #             fixed_num_devices, mem_budget, model_type, fixed_num_models,
    #             total_rate, rate_distribution,
    #             arrival_process, arrival_process_kwargs,
    #             fixed_slo, duration, policy_name))

    # run_equal_model_cases(cases, exp_name="goodput_vs_cv_scale",
    #                       output_file=args.output,
    #                       mode=args.mode, parallel=args.parallel)
