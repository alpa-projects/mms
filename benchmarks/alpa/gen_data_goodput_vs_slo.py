import argparse

from benchmarks.alpa.equal_model_case import EqualModelCase, run_equal_model_cases
from benchmarks.alpa.general_model_case import GeneralModelCase, run_general_model_cases
from alpa_serve.util import GB, batchsize_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="res_goodput_vs_slo.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--policy", type=str)
    parser.add_argument("--slo-scale", type=float)
    parser.add_argument("--trace", choices=["synthetic", "azure_v2"],
                        default="synthetic")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")
    parser.add_argument("--unequal", action="store_true")
    parser.add_argument("--model-type", type=str, default="all_transformers",
                        choices=["all_transformers", "mixed"])
    parser.add_argument("--protocol", type=str, default="http",
                        choices=["http", "ray"])
    parser.add_argument("--relax-slo", action="store_true")
    parser.add_argument("--debug-tstamp", action="store_true")
    parser.add_argument("--enable-batching", action="store_true")
    parser.add_argument("--max-batchsize", type=int, default=2)

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp",
    #           "mp-greedy-2", "mp-greedy-4", "mp-greedy-8",
    #           "mp-search",  "mp-search-sep"}
    if args.policy is not None:
        policies = [args.policy]
    else:
        # policies = ["sr-greedy", "mp-search", "sr-replace-30"]
        policies = ["mp-search"]

    if args.enable_batching:
        assert args.max_batchsize == batchsize_config[-1], f"maximum batchsize is not {args.max_batchsize}, set it in alpa_serve/util.py"
        policies = [policy + "-batch-" + str(args.max_batchsize) for policy in policies]

    exp_name = "goodput_vs_slo"
    num_devices = 8
    mem_budget = 13 * GB
    model_type = "bert-1.3b"
    num_models = 16
    total_rate = 64
    if args.trace == "synthetic":
        # choices: {"gamma", "uniform_mmpp"}
        arrival_process = "gamma"
        # choices: {"uniform", "power_law", "triangle_decay"}
        rate_distribution = "power_law"
        arrival_process_kwargs = {"cv": 4}
    elif args.trace == "azure_v2":
        # choices: {"azure_v2"}
        arrival_process = "azure_v2"
        rate_distribution = None
        arrival_process_kwargs = None

    if args.slo_scale is not None:
        slo_scales = [args.slo_scale]
    else:
        slo_scales = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14]
        # slo_scales = [14]
    duration = 1000

    if args.unequal:
        # multi-model config
        if args.model_type == "mixed":
            model_set = ["bert-1.3b", "bert-2.6b", "bert-6.7b", "moe-1.3b", "moe-2.4b", "moe-5.3b"]
        else:
            model_set = ["bert-6.7b", "moe-1.3b"]
        num_devices = 64
        total_rate = 70
        fixed_num_modelset = 8
        model_types = model_set * fixed_num_modelset
        model_names = sum([[f"{model_type}-{i}" for model_type in model_set] for i in range(fixed_num_modelset)], [])

        cases = []
        for slo_scale in slo_scales:
            for policy_name in policies:
                cases.append(GeneralModelCase(
                    exp_name, num_devices, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name))

        run_general_model_cases(cases,
                                output_file=args.output,
                                mode=args.mode,
                                debug_tstamp=args.debug_tstamp,
                                parallel=args.parallel,
                                enable_batching=args.enable_batching)
    else:
        cases = []
        for slo_scale in slo_scales:
            for policy_name in policies:
                cases.append(EqualModelCase(
                    exp_name, num_devices, mem_budget, model_type, num_models,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name,
                    None, None, None, None))


        run_equal_model_cases(cases,
                              output_file=args.output,
                              mode=args.mode,
                              relax_slo=args.relax_slo,
                              protocol=args.protocol,
                              debug_tstamp=args.debug_tstamp,
                              parallel=args.parallel,
                              enable_batching=args.enable_batching)
