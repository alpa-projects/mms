import argparse
import os
from datetime import datetime

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from alpa_serve.util import GB


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
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"],
                        default="uniform")
    parser.add_argument("--rate", type=float, default=64)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=200)

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-search"]
    mem_budget = 16 * GB
    model_type = "bert-2.6b"

    # default configuration
    fixed_num_devices = 8
    fixed_num_models = 32
    fixed_rate_scale = 1
    fixed_cv_scale = 1
    fixed_slo_scale = 1

    if args.synthetic:
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        duration = args.duration

        arrival_process = "gamma"
        arrival_process_kwargs = {"cv": args.cv}
    else:
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = -1

        arrival_process = "azure_v2"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                "cv_scale": fixed_cv_scale,
                                "trace_dir": args.trace_dir}

    num_devices_list = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    # num_models_list = [8, 16, 32, 64, 80, 96, 128]
    num_models_list = [i for i in range(16, 33, 2)]
    rate_scales = [1, 2, 4, 8, 16]
    cv_scales = [1, 2, 4, 8, 16]
    slo_scales = [0.5, 1, 2, 4]

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
                arrival_process_kwargs = {"rate_scale": num_models / fixed_num_models * 4,
                                          "cv_scale": fixed_cv_scale,
                                          "trace_dir": args.trace_dir}
                cases.append(EqualModelCase(
                    fixed_num_devices, mem_budget, model_type, num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
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

    #### goodput vs rate_scale #####
    if "goodput_vs_rate" in experiments:
        print("=== Running goodput vs. rate ===")
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

    #### goodput vs cv_scale #####
    if "goodput_vs_cv" in experiments:
        print("=== Running goodput vs. cv ===")
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

    ### model vs devices ###
    if "num_devices_vs_num_models" in experiments:
        print("=== Running #devices vs. #models ===")
        cases = []
        for num_models in num_models_list:
            for num_devices in num_devices_list:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"rate_scale": num_models / fixed_num_models,
                                              "cv_scale": fixed_cv_scale,
                                              "trace_dir": args.trace_dir}
                    cases.append(EqualModelCase(
                        num_devices, mem_budget, model_type, num_models,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

        run_equal_model_cases(cases, exp_name="num_devices_vs_num_models",
                              output_file=output_file,
                              mode=args.mode, parallel=args.parallel)
