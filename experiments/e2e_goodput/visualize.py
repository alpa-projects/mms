from alpa_serve.trace import Trace
azure_v2_trace_dir = "/home/ubuntu/efs/mms/dataset/azure_v2.pkl"
azure_v2_trace = Trace("azure_v2", azure_v2_trace_dir)
num_models = 64
model_names = [f"m{i}" for i in range(num_models)]
train_replays = azure_v2_trace.replay(model_names, model_mapping_strategy="stripe",
                                        arrival_distribution="gamma",
                                        start_time='0.0.0', end_time='1.0.0',
                                        rate_scale_factor=1,
                                        cv_scale_factor=1)
for model_name in model_names:
    train_replays[model_name].report_stats()