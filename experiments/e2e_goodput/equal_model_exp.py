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
    parser.add_argument("--exp-ids", type=str, default="0,1,2,3,4")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4"]
    fixed_slos = {"bert-1.3b": 0.5, "bert-2.6b": 0.8, "bert-6.7b": 1.2}
    mem_budget = 16 * GB
    model_type = "bert-2.6b"

    # real trace does not need these config
    rate_distribution = None
    total_rate = -1
    duration = -1

    # default configuration
    fixed_num_devices = 32
    fixed_num_models = 32
    fixed_rate_scale = 1
    fixed_cv_scale = 4
    fixed_slo_scale = fixed_slos[model_type]

    # real trace related
    arrival_process = "azure_v2"
    arrival_process_kwargs = {"rate_scale": fixed_rate_scale, 
                              "cv_scale": fixed_cv_scale,
                              "trace_dir": args.trace_dir}

    num_devices_list = [8, 16, 24, 32, 48, 64, 96, 128]
    num_models_list = [4, 8, 16, 32, 64, 80]
    rate_scales = [1, 2, 4, 8, 16]
    cv_scales = [1, 2, 4, 8, 16]
    slo_scales = [0.5, 1, 2, 4, 8]

    exp_ids = args.exp_ids.split(",")
    exp_ids = [int(exp_id) for exp_id in exp_ids]


    if args.exp_name:
        os.makedirs(args.exp_name, exist_ok=True)
        output_file = os.path.join(args.exp_name, args.output)
    else:
        output_folder = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, args.output)

    ##### goodput vs num_devices #####
    if 0 in exp_ids:
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
    if 1 in exp_ids:
        cases = []
        for num_models in num_models_list:
            for policy_name in policies:
                cases.append(EqualModelCase(
                    fixed_num_devices, mem_budget, model_type, num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name))

        run_equal_model_cases(cases, exp_name="goodput_vs_num_models",
                              output_file=output_file,
                              mode=args.mode, parallel=args.parallel)

    #### goodput vs slo #####
    if 2 in exp_ids:
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
    if 3 in exp_ids:
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
    if 4 in exp_ids:
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
