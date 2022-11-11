import argparse
from alpa_serve.profiling import ProfilingDatabase, ProfilingResult, ParallelConfig
from alpa_serve.util import GB
from gen_data_goodput import gen_uniform_mmpp_case, gen_gamma_case, run_experiment_slos


def run_experiment(experiment_name):
    policies = ["sr", "mp"]
    if experiment_name == "gamma_1":
        slos = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0]
        cases = {}
        prof_database = ProfilingDatabase("profiling_result.pkl")
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10*GB,
                    average_rate=4, cv=4, duration=100)
    elif experiment_name == "gamma_2":
        slos = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0]
        cases = {}
        prof_database = ProfilingDatabase("profiling_result.pkl")
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10*GB,
                    average_rate=4, cv=10, duration=100)
    elif experiment_name == "mmpp_1":
        prof_database = ProfilingDatabase("profiling_result.pkl")
        slos = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0]
        cases = {}
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_uniform_mmpp_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10 * GB,
                    state_durations=[1, 3], state_request_rates=[13, 1],
                    num_requests=1000)
    elif experiment_name == "gamma_2_long_slos":
        prof_database = ProfilingDatabase("profiling_result.pkl")
        slos = [1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 24.0, 28.0, 32.0, 64.0, 128.0]
        cases = {}
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10 * GB,
                    average_rate=4, cv=10, duration=100)
    elif experiment_name == "gamma_2_short_slos_no_ilp":
        prof_database = ProfilingDatabase("profiling_result.pkl")
        for pp in [2, 4]:
            del prof_database.results["bert-1.3b"].para_dict[ParallelConfig(1, 1, pp)]
        slos = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0]
        cases = {}
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10 * GB,
                    average_rate=4, cv=10, duration=400)
    elif experiment_name == "gamma_2_long_slos_no_ilp":
        prof_database = ProfilingDatabase("profiling_result.pkl")
        for pp in [2, 4]:
            del prof_database.results["bert-1.3b"].para_dict[ParallelConfig(1, 1, pp)]
        slos = [1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 24.0, 28.0, 32.0, 64.0, 128.0]
        cases = {}
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10 * GB,
                    average_rate=4, cv=10, duration=400)
    elif experiment_name.startswith("gamma_2_long_slos_no_ilp_pipeline_overhead_"):
        pipeline_overhead = float(experiment_name.split("_")[-1])
        prof_database = ProfilingDatabase(None, new_database=True)
        result = ProfilingResult("bert-1.3b", {},
                                 preprocess_cpu=0.0, postprocess_cpu=0.0)
        single_device_latency = 0.1
        single_device_weight_mem = 2.64 * GB
        for pp in [1, 8]:
            pipeline_stage_latency = single_device_latency / pp
            pipeline_weight_mem = single_device_weight_mem / pp
            if pp > 1:
                pipeline_stage_latency *= pipeline_overhead
            result.add_result(ParallelConfig(1, 1, pp),
                              batch_size=1,
                              stage_latency=[pipeline_stage_latency],
                              act_mem=[0.0],
                              weight_mem=[pipeline_weight_mem])
        prof_database.update(result)
        slos = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0]
        cases = {}
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10 * GB,
                    average_rate=4, cv=10, duration=100)
    elif experiment_name.startswith("gamma_2_long_slos_no_ilp_duration_400_pipeline_overhead_"):
        pipeline_overhead = float(experiment_name.split("_")[-1])
        prof_database = ProfilingDatabase(None, new_database=True)
        result = ProfilingResult("bert-1.3b", {},
                                 preprocess_cpu=0.0, postprocess_cpu=0.0)
        single_device_latency = 0.1
        single_device_weight_mem = 2.64 * GB
        for pp in [1, 8]:
            pipeline_stage_latency = single_device_latency / pp
            pipeline_weight_mem = single_device_weight_mem / pp
            if pp > 1:
                pipeline_stage_latency *= pipeline_overhead
            result.add_result(ParallelConfig(1, 1, pp),
                              batch_size=1,
                              stage_latency=[pipeline_stage_latency],
                              act_mem=[0.0],
                              weight_mem=[pipeline_weight_mem])
        prof_database.update(result)
        slos = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0]
        cases = {}
        for policy in policies:
            for slo in slos:
                cases[(policy, slo)] = gen_gamma_case(
                    slo, policy, prof_database,
                    num_devices=8, num_models=16, mem_budget=10 * GB,
                    average_rate=4, cv=10, duration=400)
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")
    run_experiment_slos(policies, slos, cases,
                        exp_name=experiment_name,
                        output_file=f"res_{experiment_name}.tsv",
                        parallel=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, nargs='?', default="gamma_1")
    args = parser.parse_args()
    run_experiment(args.exp_name)
