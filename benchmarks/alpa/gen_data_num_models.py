import argparse

from benchmarks.alpa.all_equal_case import AllEqualCase, run_all_equal_cases
from alpa_serve.simulator.workload import GammaProcess
from alpa_serve.util import GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_num_models.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp", "mp-greedy-2", "mp-greedy-8"}
    policies = ["sr-greedy", "mp-greedy-4", "mp-search"]
    num_devices_list = [4, 8, 12, 16]
    mem_budget = 12 * GB
    model_type = "bert-2.6b"
    num_models_list = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    per_model_rate = 2
    per_model_cv = 4
    arrival_process = GammaProcess(per_model_rate, per_model_cv)
    slos = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
    duration = 200

    cases = []
    for num_devices in num_devices_list:
        for slo in slos:
            for policy in policies:
                for num_models in num_models_list:
                    cases.append(AllEqualCase(
                        num_devices, mem_budget, model_type, num_models,
                        arrival_process, slo, duration, policy))

    run_all_equal_cases(cases,
                        exp_name=args.exp_name,
                        output_file=args.output,
                        mode=args.mode,
                        parallel=args.parallel)
