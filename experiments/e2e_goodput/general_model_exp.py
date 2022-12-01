import argparse
import datetime
import os

from benchmarks.alpa.general_model_case import GeneralModelCase, run_general_model_cases
from alpa_serve.util import GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_general_model_cases.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")
    parser.add_argument("--trace-dir", type=str, default="~/azure_v2.pkl")
    parser.add_argument("--exp-ids", type=str, default="0,1,2,3,4")
    parser.add_argument("--mixed", action="store_true")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4"]
    mem_budget = 16 * GB
    
    # default config
    fixed_num_devices = 32
    fixed_rate_scale = 1
    fixed_cv_scale = 4
    fixed_slo_scale = 1
    fixed_num_modelset = 10

    # multi-model config
    if args.mixed:
        model_types = ["bert-1.3b", "bert-2.6b", "bert-6.7b", "moe-1.3b", "moe-2.4b", "moe-7.1b"] * fixed_num_modelset
        model_names = sum([[f"t_s{i}", f"t_m{i}", f"t_b{i}", f"m_s{i}", f"m_m{i}", f"m_b{i}"] for i in range(fixed_num_modelset)], [])
    else:
        model_types = ["bert-1.3b", "bert-2.6b", "bert-6.7b"] * fixed_num_modelset
        model_names = sum([[f"t_s{i}", f"t_m{i}", f"t_b{i}"] for i in range(fixed_num_modelset)], [])


    # real trace does not need these config
    rate_distribution = None
    total_rate = -1
    duration = -1

    # real trace config
    arrival_process = "azure_v2"
    arrival_process_kwargs = {"rate_scale": fixed_rate_scale, 
                              "cv_scale": fixed_cv_scale,
                              "trace_dir": args.trace_dir}

    num_devices_list = [16, 24, 32, 48, 64, 96, 128]
    num_modelset_list = [1, 4, 8, 10, 12, 14]
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
                cases.append(GeneralModelCase(
                    num_devices, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name))

        run_general_model_cases(cases, exp_name="goodput_vs_num_devices",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)

    ##### goodput vs slo #####
    if 1 in exp_ids:
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

    ##### goodput vs rate_scale #####
    if 2 in exp_ids:
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


    ##### goodput vs cv_scale #####
    if 3 in exp_ids:
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


    ##### goodput vs num_models #####
    if 4 in exp_ids:
        cases = []
        for num_modelset in num_modelset_list:
            for policy_name in policies:
                new_model_types = ["bert-1.3b", "bert-2.6b", "bert-6.7b"] * num_modelset
                new_model_names = sum([[f"t_s{i}", f"t_m{i}", f"t_b{i}"] for i in range(num_modelset)], [])
                cases.append(GeneralModelCase(
                    fixed_num_devices, mem_budget, new_model_types, new_model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name))

        run_general_model_cases(cases, exp_name="goodput_vs_num_models",
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel)
