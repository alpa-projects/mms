import argparse
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.util import GB
from gen_data_goodput import gen_gamma_case, run_experiment_slos


def run_experiment(experiment_name):
    prof_database = ProfilingDatabase("profiling_result.pkl")

    policies = ["sr", "mp"]
    if experiment_name == "gamma_1":
        slos = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0]
        cases = {}
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10*GB,
                    average_rate=4, cv=4, duration=100)
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")
    run_experiment_slos(policies, slos, cases,
                        exp_name=experiment_name,
                        output_file=f"res_{experiment_name}.tsv",
                        parallel=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp-name", type=str, nargs='?', default="gamma_1")
    args = parser.parse_args()
    run_experiment(args.exp_name)
