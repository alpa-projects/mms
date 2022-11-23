from alpa_serve.trace import Trace, TraceReplay

trace_name = "azure_v2"
trace_dir = "azure_v2.pkl"


n_model = 10
models = [f"gpt{i}" for i in range(n_model)]
trace = Trace(trace_name, trace_dir)

replays = trace.replay_vanilla(models,
                               model_mapping_strategy="stripe",
                               start_time="0.0.0",
                               end_time="1.0.0")
for m in replays:
    replays[m].report_stats()
    replays[m].visualize()

interval_seconds = [600, 1800, 3600]
for interval_secs in interval_seconds:
    for distribution in ["gamma"]:
    # for distribution in ["gamma"]:
        replays = trace.replay(models,
                               model_mapping_strategy="stripe",
                               start_time="0.0.0",
                               end_time="1.0.0",
                               arrival_distribution=distribution,
                               interval_seconds=interval_secs)
        for m in replays:
            replays[m].report_stats()
            replays[m].visualize()
