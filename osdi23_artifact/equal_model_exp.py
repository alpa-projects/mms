import argparse
import os
from datetime import datetime

from alpa_serve.util import GB
from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from equal_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--output", type=str, default="res_various_metrics.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")
    parser.add_argument("--trace-dir", type=str, default="~/azure_v2.pkl")
    parser.add_argument("--exp-ids", type=str, default="all",
                        choices=["all", "goodput_vs_num_devices", "goodput_vs_num_models",
                              "goodput_vs_slo", "goodput_vs_rate", "goodput_vs_cv",
                              "num_devices_vs_num_models"])
    parser.add_argument("--model-type", type=str, default="bert-1.3b",
                        choices=["bert-1.3b", "bert-2.6b", "bert-6.7b", "bert-103.5b"])
    parser.add_argument("--mem-budget", type=int, default=13)
    parser.add_argument("--workload", type=str, default="synthetic",
                        choices=["synthetic", "azure_v1", "azure_v2"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"],
                        default="power_law")
    parser.add_argument("--rate", type=float, default=64)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=200)
    parser.add_argument('--enable-batching', action='store_true')
    parser.add_argument("--large-models", action="store_true")

    args = parser.parse_args()

    model_type = args.model_type
    mem_budget = args.mem_budget * GB

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    if model_type == "bert-103.5b":
        policies = ["mp-equal-16-1", "mp-equal-8-2", "mp-equal-4-4", "mp-equal-2-8", "mp-search"]
    else:
        if args.workload == "azure_v2":
            policies = ["sr-greedy", "sr-replace-21600", "mp-search"]
        else:
            policies = ["sr-greedy", "sr-replace-30", "mp-search"]
    if args.enable_batching:
        policies = [policy + "-batch" for policy in policies]

    # workload config
    if args.workload == "synthetic":
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        duration = args.duration

        arrival_process = "gamma"
        arrival_process_kwargs = {"cv": args.cv}

        fixed_num_devices, fixed_num_models, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_models_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = synthetic_suite[model_type]
    elif args.workload == "azure_v1":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = -1

        fixed_num_devices, fixed_num_models, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_models_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v1_suite[model_type]

        arrival_process = "azure_v1"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": args.trace_dir}
    elif args.workload == "azure_v2":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = -1

        fixed_num_devices, fixed_num_models, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_models_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v2_suite[model_type]

        arrival_process = "azure_v2"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": args.trace_dir}
    else:
        raise ValueError("Unsupported workload!")

    # output file
    if args.output.endswith(".tsv"):
        output_file_name = args.output
    else:
        output_file_name = args.output + ".tsv"

    if args.exp_name:
        os.makedirs(args.exp_name, exist_ok=True)
        output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   args.exp_name, output_file_name)
    else:
        output_folder = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   output_folder, output_file_name)

    # parse exp ids:
    if args.exp_ids == "all":
        experiments = ["goodput_vs_num_devices", "goodput_vs_num_models", "goodput_vs_slo",
                       "goodput_vs_rate", "goodput_vs_cv", "num_devices_vs_num_models"]
    else:
        assert args.exp_ids in ["goodput_vs_num_devices", "goodput_vs_num_models", "goodput_vs_slo",
                                "goodput_vs_rate", "goodput_vs_cv", "num_devices_vs_num_models"]
        experiments = [args.exp_ids]

    if args.large_models:
        experiments = ["goodput_vs_rate", "goodput_vs_cv", "goodput_vs_slo"]

    cases = []
    ##### goodput vs num_devices #####
    if "goodput_vs_num_devices" in experiments:
        print("=== Running goodput vs. #devices ===")
        exp_name = "goodput_vs_num_devices"
        for num_devices in num_devices_list:
            for policy_name in policies:
                cases.append(EqualModelCase(exp_name,
                    num_devices, mem_budget, model_type, fixed_num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name, None, None, None, None))

    # #### goodput vs num_models #####
    # if "goodput_vs_num_models" in experiments:
    #     print("=== Running goodput vs. #models ===")
    #     exp_name = "goodput_vs_num_models"
    #     for num_models in num_models_list:
    #         for policy_name in policies:
    #             # Note(Hao): we need to scale the rate as well to keep the total traffic unchanged.
    #             # when num_model = fix_num_models / 4, the total cluster rate is 1 q/s.
    #             if args.workload == "synthetic":
    #                 cases.append(EqualModelCase(exp_name,
    #                     fixed_num_devices, mem_budget, model_type, num_models,
    #                     total_rate * num_models / fixed_num_models, rate_distribution,
    #                     arrival_process, arrival_process_kwargs,
    #                     fixed_slo_scale, duration, policy_name, None, None, None, None))
    #             else:
    #                 scale_factor = num_models / fixed_num_models
    #                 new_arrival_process_kwargs = {"rate_scale": scale_factor * fixed_rate_scale,
    #                                               "cv_scale": fixed_cv_scale,
    #                                               "trace_dir": args.trace_dir}
    #                 cases.append(EqualModelCase(exp_name,
    #                     fixed_num_devices, mem_budget, model_type, num_models,
    #                     total_rate, rate_distribution,
    #                     arrival_process, new_arrival_process_kwargs,
    #                     fixed_slo_scale, duration, policy_name, None, None, None, None))

    #### goodput vs slo #####
    if "goodput_vs_slo" in experiments:
        print("=== Running goodput vs. SLO ===")
        exp_name = "goodput_vs_slo"
        for slo_scale in slo_scales:
            for policy_name in policies:
                cases.append(EqualModelCase(exp_name,
                    fixed_num_devices, mem_budget, model_type, fixed_num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name, None, None, None, None))

    #### goodput vs rate/rate_scale #####
    if "goodput_vs_rate" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. rate ===")
            exp_name = "goodput_vs_rate"
            for new_rate in rate_list:
                for policy_name in policies:
                    cases.append(EqualModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        new_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name, None, None, None, None))
        else:
            print("=== Running goodput vs. rate_scale ===")
            exp_name = "goodput_vs_rate_scale"
            for rate_scale in rate_scales:
                for policy_name in policies:
                    arrival_process_kwargs = {"rate_scale": rate_scale,
                                            "cv_scale": fixed_cv_scale,
                                            "trace_dir": args.trace_dir}
                    cases.append(EqualModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name, None, None, None, None))


    #### goodput vs cv/cv_scale #####
    if "goodput_vs_cv" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. cv ===")
            exp_name = "goodput_vs_cv"
            for new_cv in cv_list:
                for policy_name in policies:
                    arrival_process_kwargs = {"cv": new_cv}
                    cases.append(EqualModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name, None, None, None, None))
        else:
            print("=== Running goodput vs. cv_scale ===")
            exp_name = "goodput_vs_cv_scale"
            for cv_scale in cv_scales:
                for policy_name in policies:
                    arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                              "cv_scale": cv_scale,
                                              "trace_dir": args.trace_dir}
                    cases.append(EqualModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name, None, None, None, None))

    n_cases = len(cases)
    M = 8
    n_case_each_run = (n_cases + M - 1) // M
    for i in range(M):
        start_case = i * n_case_each_run
        end_case = (i + 1) * n_case_each_run  if (i + 1) * n_case_each_run < n_cases else n_cases
        run_equal_model_cases(cases[start_case:end_case],
                              output_file=output_file,
                              mode=args.mode, parallel=args.parallel,
                              enable_batching=args.enable_batching)
    # ### model vs devices ###
    # if "num_devices_vs_num_models" in experiments:
    #     print("=== Running #devices vs. #models ===")
    #     cases = []

    #     # We need more data points to generate a meaningful curve.
    #     num_devices_list = [i for i in range(2, 65, 4)]
    #     num_models_list = [i for i in range(8, 97, 2)]

    #     for num_models in num_models_list:
    #         for num_devices in num_devices_list:
    #             if "1.3b" in model_type or "2.6b" in model_type:
    #                 if num_devices * 2 > num_models:
    #                     print(f"Skip the case num_devices = {num_devices} and num_models = {num_models} "
    #                           f"because the goodput will likely be 100%.")
    #                     continue
    #             model_size = float(model_type.split("-")[-1][:-1]) * 2
    #             if mem_budget / GB * num_devices < model_size * num_models * 2 / 3:
    #                 print(f"Skip the case num_devices = {num_devices} and num_models = {num_models} "
    #                       f"because the goodput will be less than 66%.")

    #             for policy_name in policies:
    #                 new_arrival_process_kwargs = {"rate_scale": num_models / fixed_num_models * 4,
    #                                           "cv_scale": fixed_cv_scale,
    #                                           "trace_dir": args.trace_dir}
    #                 cases.append(EqualModelCase(
    #                     num_devices, mem_budget, model_type, num_models,
    #                     total_rate, rate_distribution,
    #                     arrival_process, new_arrival_process_kwargs,
    #                     fixed_slo_scale, duration, policy_name))

    #     run_equal_model_cases(cases, exp_name="num_devices_vs_num_models",
    #                           output_file=output_file,
    #                           mode=args.mode, parallel=args.parallel)
