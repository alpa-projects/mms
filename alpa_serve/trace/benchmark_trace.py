from alpa_serve.trace import Trace, TraceReplay, report_group_stats
from scipy.stats import entropy
import numpy as np


# trace_name = "azure_v2"
# trace_dir = "/mnt/e/projects/projects/dataset/mms_dataset/azure_v2.pkl"

trace_name = "azure_v1"
trace_dir = "/mnt/e/projects/projects/dataset/mms_dataset/azure_v1.pkl"

n_model = 48
models = [f"gpt{i}" for i in range(n_model)]
trace = Trace(trace_name, trace_dir)


def cdf(x):
    x = np.array(x)
    return np.cumsum(np.sort(x)[::-1]) / np.sum(x)


entropies = {}

for day in range(14):
    for h in range(24):
        start_time = str(day) + "." + str(h) + ".0"
        if h < 23:
            end_time = str(day) + "." + str(h+1) + ".0"
        else:
            end_time = str(day) + "." + "23.60"


        replays = trace.replay(models,
                               model_mapping_strategy="stripe",
                               start_time=start_time,
                               end_time=end_time,
                               interval_seconds=60,
                               arrival_distribution="exponential",
                               rate_scale_factor=1e-3)
        # for m in replays:
        #     replays[m].report_stats()
            # replays[m].visualize(n_interval=1000)
        report_group_stats(list(replays.values()))
        x = [replays[model].arrivals.size for model in replays]
        # print(x)
        entropies[start_time] = entropy(x)
        print(f"Entropy for {start_time} - {end_time}: {entropy(x)}, top-5 {np.sum(cdf(x)[:5]) / np.sum(cdf(x))}")

print(entropies)
print(max(entropies.values()))

# for day in range(13, 14):
#     start_time = str(day) + ".0.0"
#     end_time = str(day+1) + ".0.0"
#
#     if day == 13:
#         end_time = "13.23.60"
#     # replication_factors = [1, 2, 3]
#     print(f"Day: {start_time} - {end_time}")
#     distributions = ["gamma"]
    # for rf in replication_factors:
    # replays = trace.replay_vanilla(models,
    #                                model_mapping_strategy="stripe",
    #                                start_time=start_time,
    #                                end_time=end_time)
    # for m in replays:
    #     replays[m].report_stats()
    #     # replays[m].visualize(n_interval=1000)
    # report_group_stats(list(replays.values()))

    # for distribution in distributions:
    #     replays = trace.replay(models,
    #                            model_mapping_strategy="stripe",
    #                            start_time=start_time,
    #                            end_time=end_time,
    #                            interval_seconds=5400,
    #                            arrival_distribution=distribution)
    #     # for m in replays:
    #     #     replays[m].report_stats()
    #         # replays[m].visualize(n_interval=1000)
    #     report_group_stats(list(replays.values()))
    #     x = [replays[model].arrivals.size for model in replays]
    #     print(x)
    #     print(f"Entropy for {start_time} - {end_time}: {entropy(x)}, CDF: {cdf(x)}")

# replays = trace.replay(models,
#                        model_mapping_strategy="stripe",
#                        start_time="0.0.0",
#                        end_time="2.0.0",
#                        interval_seconds=86400 // 2,
#                        arrival_distribution="gamma")
# for m in replays:
#     replays[m].report_stats()
#     replays[m].visualize()


# interval_seconds = [600]
# time_scale_factors = [2.0, 4.0, 8.0]
# for interval_secs in interval_seconds:
#     # for distribution in ["exponential", "gamma", "vanilla"]:
#     for distribution in ["vanilla"]:
#         for time_scale_factor in time_scale_factors:
#             replays = trace.replay(models,
#                                    model_mapping_strategy="stripe",
#                                    start_time="0.0.0",
#                                    end_time="1.0.0",
#                                    arrival_distribution=distribution,
#                                    interval_seconds=interval_secs,
#                                    time_scale_factor=time_scale_factor)
#             for m in replays:
#                 replays[m].report_stats()
#                 replays[m].visualize()
