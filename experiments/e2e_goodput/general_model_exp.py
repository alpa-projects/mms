import argparse
import datetime
import os

from benchmarks.alpa.general_model_case import GeneralModelCase, run_general_model_cases
from alpa_serve.util import GB
from general_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_general_model_cases.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")
    parser.add_argument("--trace-dir", type=str, default="~/azure_v2.pkl")
    parser.add_argument("--exp-ids", type=str, default="all",
                        choices=["all", "goodput_vs_num_devices", "goodput_vs_num_models",
                              "goodput_vs_slo", "goodput_vs_rate", "goodput_vs_cv",
                              "device_vs_model"])
    parser.add_argument("--mem_budget", type=int, default=14)
    parser.add_argument("--workload", type=str, default="synthetic",
                        choices=["synthetic", "azure_v1", "azure_v2"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"],
                        default="power_law")
    parser.add_argument("--rate", type=float, default=64)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=200)
    parser.add_argument("--model_type", type=str, default="all_transformers",
                        choices=["all_transformers", "mixed"])

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8", "mp-search"}
    policies = ["sr-greedy", "mp-search"]
    mem_budget = args.mem_budget * GB
    model_type = args.model_type
    
    # default config
    if args.model_type == "mixed":
        fixed_num_devices = 64
        fixed_num_modelset = 10
    else:
        fixed_num_devices = 32
        fixed_num_modelset = 10

    fixed_rate_scale = 1
    fixed_cv_scale = 1
    fixed_slo_scale = 5

    # multi-model config
    if args.model_type == "mixed":
        model_set = ["bert-1.3b", "bert-2.6b", "bert-6.7b", "moe-1.3b", "moe-2.4b", "moe-7.1b"]
    else:
        model_set = ["bert-1.3b", "bert-2.6b", "bert-6.7b"]
    
    model_types = model_set * fixed_num_modelset
    model_names = sum([[f"{model_type}-{i}" for model_type in model_set] for i in range(fixed_num_modelset)], [])

    # workload config
    if args.workload == "synthetic":
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        duration = args.duration

        arrival_process = "gamma"
        arrival_process_kwargs = {"cv": args.cv}

        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = synthetic_suite[model_type]
    elif args.workload == "azure_v2":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = -1

        arrival_process = "azure_v2"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": args.trace_dir}
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v2_suite[model_type]
    else:
        raise ValueError("Unsupported workload!")

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
                       "goodput_vs_rate", "goodput_vs_cv", "device_vs_model"]
    else:
        assert args.exp_ids in ["goodput_vs_num_devices", "goodput_vs_num_models", "goodput_vs_slo",
                       "goodput_vs_rate", "goodput_vs_cv", "device_vs_model"]
        experiments = [args.exp_ids]


    ##### goodput vs num_devices #####
    if "goodput_vs_num_devices" in experiments:
        print("=== Running goodput vs. #devices ===")
        cases = []
        for num_devices in num_devices_list:
            for policy_name in policies:
                cases.append(GeneralModelCase(
                    num_devices, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name))

        run_general_model_cases(cases, exp_name="goodput_vs_num_devices",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)
    
    ##### goodput vs num_models #####
    if "goodput_vs_num_models" in experiments:
        print("=== Running goodput vs. #models ===")
        cases = []
        for num_modelset in num_modelset_list:
            for policy_name in policies:
                new_model_types = model_set * num_modelset
                new_model_names = sum([[f"{model_type}-{i}" for model_type in model_set] for i in range(num_modelset)], [])
                if args.workload == "synthetic":
                     cases.append(GeneralModelCase(
                        fixed_num_devices, mem_budget, new_model_types, new_model_names,
                        total_rate * num_modelset / fixed_num_modelset, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))
                else:
                    new_arrival_process_kwargs = {"rate_scale": num_modelset / fixed_num_modelset,
                                                "cv_scale": fixed_cv_scale,
                                                "trace_dir": args.trace_dir}
                    cases.append(GeneralModelCase(
                        fixed_num_devices, mem_budget, new_model_types, new_model_names,
                        total_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

        run_general_model_cases(cases, exp_name="goodput_vs_num_models",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)


    ##### goodput vs slo #####
    if "goodput_vs_slo" in experiments:
        print("=== Running goodput vs. SLO ===")
        cases = []
        for slo_scale in slo_scales:
            for policy_name in policies:
                cases.append(GeneralModelCase(
                    fixed_num_devices, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name))

        run_general_model_cases(cases, exp_name="goodput_vs_slo",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)

    ##### goodput vs rate/rate_scale #####
    if "goodput_vs_rate" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. rate ===")
            cases = []
            for new_rate in rate_list:
                for policy_name in policies:
                    cases.append(GeneralModelCase(
                        fixed_num_devices, mem_budget, model_types, model_names,
                        new_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))
            run_general_model_cases(cases, exp_name="goodput_vs_rate",
                                    output_file=output_file,
                                    mode=args.mode, parallel=args.parallel)
        else:
            print("=== Running goodput vs. rate_scale ===")
            cases = []
            for rate_scale in rate_scales:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"rate_scale": rate_scale,
                                                  "cv_scale": fixed_cv_scale,
                                                  "trace_dir": args.trace_dir}
                    cases.append(GeneralModelCase(
                        fixed_num_devices, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

            run_general_model_cases(cases, exp_name="goodput_vs_rate_scale",
                                    output_file=output_file,
                                    mode=args.mode, parallel=args.parallel)


    ##### goodput vs cv/cv_scale #####
    if "goodput_vs_cv" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. cv ===")
            cases = []
            for new_cv in cv_list:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"cv": new_cv}
                    cases.append(GeneralModelCase(
                        fixed_num_devices, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

            run_general_model_cases(cases, exp_name="goodput_vs_cv",
                                    output_file=output_file,
                                    mode=args.mode, parallel=args.parallel)
        else:
            print("=== Running goodput vs. cv_scale ===")
            cases = []
            for cv_scale in cv_scales:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                                "cv_scale": cv_scale,
                                                "trace_dir": args.trace_dir}
                    cases.append(GeneralModelCase(
                        fixed_num_devices, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

            run_general_model_cases(cases, exp_name="goodput_vs_cv_scale",
                                    output_file=output_file,
                                    mode=args.mode, parallel=args.parallel)
