import argparse

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

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4"]
    mem_budget = 16 * GB
    
    # multi-model config
    small_transformer_nums = 10
    middle_transformer_nums = 10
    big_transformer_nums = 10
    # default config
    fixed_num_devices = 32
    fixed_rate_scale = 1
    fixed_cv_scale = 4
    fixed_slo_scale = 1
    fixed_num_modelset = 10
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
    num_modelset_list = [10, 12, 14]
    rate_scales = [1, 2, 4, 8, 16]
    cv_scales = [1, 2, 4, 8, 16]
    slo_scales = [0.5, 1, 2, 4, 8]

    ##### goodput vs num_devices #####
    cases = []
    for num_devices in num_devices_list:
        for policy_name in policies:
            cases.append(GeneralModelCase(
                num_devices, mem_budget, model_types, model_names,
                total_rate, rate_distribution,
                arrival_process, arrival_process_kwargs,
                fixed_slo_scale, duration, policy_name))

    run_general_model_cases(cases, exp_name="goodput_vs_num_devices",
                            output_file=args.output,
                            mode=args.mode, parallel=args.parallel)

    ##### goodput vs slo #####
    cases = []
    for slo_scale in slo_scales:
        for policy_name in policies:
            cases.append(GeneralModelCase(
                fixed_num_devices, mem_budget, model_types, model_names,
                total_rate, rate_distribution,
                arrival_process, arrival_process_kwargs,
                slo_scale, duration, policy_name))

    run_general_model_cases(cases, exp_name="goodput_vs_slo",
                            output_file=args.output,
                            mode=args.mode, parallel=args.parallel)

    ##### goodput vs rate_scale #####
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
                            output_file=args.output,
                            mode=args.mode, parallel=args.parallel)


    ##### goodput vs cv_scale #####
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
                            output_file=args.output,
                            mode=args.mode, parallel=args.parallel)


    ##### goodput vs num_models #####
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
                            output_file=args.output,
                            mode=args.mode, parallel=args.parallel)

