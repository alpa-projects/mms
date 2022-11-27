from alpa_serve.trace import Trace, TraceReplay

trace_name = "azure_v2"
trace_dir = "azure_v2.pkl"


n_model = 32
models = [f"gpt{i}" for i in range(n_model)]
trace = Trace(trace_name, trace_dir)


# replication_factors = [1, 2, 3]

# distributions = ["gamma", "loggamma", "pareto"]
# # for rf in replication_factors:
# replays = trace.replay(models,
#                        model_mapping_strategy="stripe",
#                        start_time="5.0.0",
#                        end_time="6.0.0",
#                        arrival_distribution="v")
# for m in replays:
#     replays[m].report_stats()
#     replays[m].visualize(n_interval=1000)

# for distribution in distributions:
replays = trace.replay(models,
                       model_mapping_strategy="stripe",
                       start_time="6.0.0",
                       end_time="7.0.0",
                       interval_seconds=7200,
                       arrival_distribution="gamma")
for m in replays:
    replays[m].report_stats()
    replays[m].visualize(n_interval=1000)

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
