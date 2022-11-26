from alpa_serve.trace import Trace, TraceReplay

trace_name = "azure_v2"
trace_dir = "azure_v2.pkl"


n_model = 3
models = [f"gpt{i}" for i in range(n_model)]
trace = Trace(trace_name, trace_dir)


replication_factors = [1, 2, 3]

for rf in replication_factors:
    replays = trace.replay(models,
                           model_mapping_strategy="stripe",
                           start_time="0.0.0",
                           end_time="1.0.0",
                           arrival_distribution="vanilla",
                           replication_factor=rf)
    for m in replays:
        replays[m].report_stats()
        replays[m].visualize()

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
