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
                        choices=["bert-1.3b", "bert-2.6b", "bert-6.7b"])
    parser.add_argument("--mem-budget", type=int, default=14)
    parser.add_argument("--workload", type=str, default="synthetic",
                        choices=["synthetic", "azure_v1", "azure_v2"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"],
                        default="power_law")
    parser.add_argument("--rate", type=float, default=64)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=200)

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-search"]
    mem_budget = args.mem_budget * GB
    model_type = args.model_type

    # default config
    fixed_num_devices = 32
    fixed_num_models = 32
    fixed_rate_scale = 1
    fixed_cv_scale = 1
    fixed_slo_scale = 5

    # workload config
    if args.workload == "synthetic":
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        duration = args.duration

        arrival_process = "gamma"
        arrival_process_kwargs = {"cv": args.cv}

        num_devices_list, num_models_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = synthetic_suite[model_type]
    elif args.workload == "azure_v1":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = -1

        arrival_process = "azure_v1"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": args.trace_dir}
        num_devices_list, num_models_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v1_suite[model_type]
    elif args.workload == "azure_v2":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = -1

        arrival_process = "azure_v2"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": args.trace_dir}
        if model_type == "bert-1.3b":
            fixed_num_devices = 8
            fixed_num_models = 48
        elif model_type == "bert-2.6b":
            fixed_num_devices = 16
            fixed_num_models = 48
        else:
            fixed_num_devices = 48
            fixed_num_models = 48
        num_devices_list, num_models_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v2_suite[model_type]
    else:
        raise ValueError("Unsupported workload!")

    # output file
    if args.output.endswith(".tsv"):
        output_file_name = args.output
    else:
        output_file_name = args.output + ".tsv"

    if args.exp_name:
        os.makedirs(args.exp_name, exist_ok=True)
        output_file = os.path.join(args.exp_name, output_file_name)
    else:
        output_folder = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, output_file_name)

    # parse exp ids:
    if args.exp_ids == "all":
        experiments = ["goodput_vs_num_devices", "goodput_vs_num_models", "goodput_vs_slo",
                       "goodput_vs_rate", "goodput_vs_cv", "num_devices_vs_num_models"]
    else:
        assert args.exp_ids in ["goodput_vs_num_devices", "goodput_vs_num_models", "goodput_vs_slo",
                                "goodput_vs_rate", "goodput_vs_cv", "num_devices_vs_num_models"]
        experiments = [args.exp_ids]


    ##### goodput vs num_devices #####
    if "goodput_vs_num_devices" in experiments:
        print("=== Running goodput vs. #devices ===")
        cases = []
        for num_devices in num_devices_list:
            for policy_name in policies:
                cases.append(EqualModelCase(
                    num_devices, mem_budget, model_type, fixed_num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name))

        run_equal_model_cases(cases, exp_name="goodput_vs_num_devices",
                              output_file=output_file,
                              mode=args.mode, parallel=args.parallel)

    #### goodput vs num_models #####
    if "goodput_vs_num_models" in experiments:
        print("=== Running goodput vs. #models ===")
        cases = []
        for num_models in num_models_list:
            for policy_name in policies:
                # Note(Hao): we need to scale the rate as well to keep the total traffic unchanged.
                # when num_model = fix_num_models / 4, the total cluster rate is 1 q/s.
                if args.workload == "synthetic":
                    cases.append(EqualModelCase(
                        fixed_num_devices, mem_budget, model_type, num_models,
                        total_rate * num_models / fixed_num_models, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))
                else:
                    scale_factor = num_models / fixed_num_models
                    new_arrival_process_kwargs = {"rate_scale": scale_factor * fixed_rate_scale,
                                                  "cv_scale": fixed_cv_scale,
                                                  "trace_dir": args.trace_dir}
                    cases.append(EqualModelCase(
                        fixed_num_devices, mem_budget, model_type, num_models,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

        run_equal_model_cases(cases, exp_name="goodput_vs_num_models",
                              output_file=output_file,
                              mode=args.mode, parallel=args.parallel)

    #### goodput vs slo #####
    if "goodput_vs_slo" in experiments:
        print("=== Running goodput vs. SLO ===")
        cases = []
        for slo_scale in slo_scales:
            for policy_name in policies:
                cases.append(EqualModelCase(
                    fixed_num_devices, mem_budget, model_type, fixed_num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name))

        run_equal_model_cases(cases, exp_name="goodput_vs_slo",
                              output_file=output_file,
                              mode=args.mode, parallel=args.parallel)

    #### goodput vs rate/rate_scale #####
    if "goodput_vs_rate" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. rate ===")
            cases = []
            for new_rate in rate_list:
                for policy_name in policies:
                    cases.append(EqualModelCase(
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        new_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

            run_equal_model_cases(cases, exp_name="goodput_vs_rate",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)
        else:
            print("=== Running goodput vs. rate_scale ===")
            cases = []
            for rate_scale in rate_scales:
                for policy_name in policies:
                    arrival_process_kwargs = {"rate_scale": rate_scale,
                                            "cv_scale": fixed_cv_scale,
                                            "trace_dir": args.trace_dir}
                    cases.append(EqualModelCase(
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

            run_equal_model_cases(cases, exp_name="goodput_vs_rate_scale",
                                  output_file=output_file,
                                  mode=args.mode, parallel=args.parallel)

    #### goodput vs cv/cv_scale #####
    if "goodput_vs_cv" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. cv ===")
            cases = []
            for new_cv in cv_list:
                for policy_name in policies:
                    arrival_process_kwargs = {"cv": new_cv}
                    cases.append(EqualModelCase(
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

            run_equal_model_cases(cases, exp_name="goodput_vs_cv",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)
        else:
            print("=== Running goodput vs. cv_scale ===")
            cases = []
            for cv_scale in cv_scales:
                for policy_name in policies:
                    arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                              "cv_scale": cv_scale,
                                              "trace_dir": args.trace_dir}
                    cases.append(EqualModelCase(
                        fixed_num_devices, mem_budget, model_type, fixed_num_models,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

            run_equal_model_cases(cases, exp_name="goodput_vs_cv_scale",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)

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
