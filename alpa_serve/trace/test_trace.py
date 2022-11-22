import pytest

import numpy as np

from alpa_serve.trace import Trace, load_azure_v1_trace, load_azure_v2_trace

azure_v1_trace_dir = "/mnt/e/projects/projects/mms/dataset/azurefunctions-dataset2019"
azure_v2_trace_dir = "/mnt/e/projects/projects/mms/dataset/AzureFunctionsInvocationTraceForTwoWeeksJan2021"

azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
# azure_v1_trace = None
azure_v2_trace = None
# azure_v2_trace = Trace("azure_v2", azure_v2_trace_dir)
models = [f"gpt-{i}" for i in range(30)]


def get_trace(trace_name):
    if trace_name == "azure_v1":
        trace = azure_v1_trace
        if trace is None:
            trace = Trace("azure_v1", azure_v1_trace_dir)
    else:
        trace = azure_v2_trace
        if trace is None:
            trace = Trace("azure_v2", azure_v2_trace_dir)
    return trace


@pytest.mark.parametrize("n_day", [1, 2, 4, 8, 14])
@pytest.mark.skip(reason="slow test")
def test_read_azure_v1(n_day):
    load_azure_v1_trace(azure_v1_trace_dir, n_day)

@pytest.mark.skip(reason="too slow")
def test_read_azure_v2():
    load_azure_v2_trace(azure_v2_trace_dir)

@pytest.mark.skip(reason="tested")
@pytest.mark.parametrize("start_time, end_time",
                         [("0.0.0", "1.0.0"),
                          ("0.0.0", "0.23.60"),
                          ("0.0.0", "7.0.0"),
                          ("0.0.0", "13.23.60"),
                          ("5.5.5", "8.8.8"),
                          ("11.0.0", "13.23.60"),
                          ("12.23.60", "13.23.60"),
                          ("13.0.0", "13.23.60")])
def test_slice_azure_v2_trace(start_time, end_time):
    trace = get_trace("azure_v2")
    arrivals = trace.slice(start_time=start_time, end_time=end_time)
    Trace.report_stats(arrivals)
    start_d, start_h, start_m = trace.timestr_to_dhm(start_time)
    end_d, end_h, end_m = trace.timestr_to_dhm(end_time)
    start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
    end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
    for function, arrival in arrivals.items():
        np.all(arrival >= start_timestamp_seconds)
        np.all(arrival < end_timestamp_seconds)

@pytest.mark.skip(reason="tested")
@pytest.mark.parametrize("start_time, end_time",
                         [("0.0.0", "1.0.0"),
                          ("0.0.0", "0.23.60"),
                          ("0.0.0", "7.0.0"),
                          ("0.0.0", "13.23.60"),
                          ("5.5.5", "8.8.8"),
                          ("11.0.0", "13.23.60"),
                          ("12.23.60", "13.23.60"),
                          ("13.0.0", "13.23.60")])
def test_slice_azure_v1_trace(start_time, end_time):
    trace = get_trace("azure_v1")
    histogram = trace.slice(start_time, end_time)
    Trace.report_stats(histogram)

    start_d, start_h, start_m = trace.timestr_to_dhm(start_time)
    end_d, end_h, end_m = trace.timestr_to_dhm(end_time)
    start_slot = start_d * 24 * 60 + start_h * 60 + start_m
    end_slot = end_d * 24 * 60 + end_h * 60 + end_m
    n_slot = end_slot - start_slot
    for function, h in histogram.items():
        assert h.size == n_slot

@pytest.mark.parametrize("start_time, end_time",
                         [("0.0.0", "1.0.0"),
                          ("0.0.0", "0.23.60"),
                          ("0.0.0", "7.0.0"),
                          ("0.0.0", "13.23.60"),
                          ("5.5.5", "8.8.8"),
                          ("11.0.0", "13.23.60"),
                          ("12.23.60", "13.23.60"),
                          ("13.0.0", "13.23.60")])
def test_replay_vanilla(start_time, end_time):
    trace_replays = azure_v2_trace.replay_vanilla(models, start_time=start_time,
                                                  end_time=end_time)
    for model, replay in trace_replays.items():
        replay.report_stats()
        replay.visualize()


@pytest.mark.parametrize("start_time, end_time",
                         [("0.0.0", "1.0.0"),
                          ("0.0.0", "0.23.60"),
                          ("0.0.0", "7.0.0"),
                          ("0.0.0", "13.23.60"),
                          ("5.5.5", "8.8.8"),
                          ("11.0.0", "13.23.60"),
                          ("12.23.60", "13.23.60"),
                          ("13.0.0", "13.23.60")])
@pytest.mark.parametrize("model_mapping_strategy", ["round_robin", "stripe"])
@pytest.mark.parametrize("arrival_distribution", ["exponential", "gamma"])
@pytest.mark.parametrize("interval_seconds", [60, 1800, 3600, 14400])
def test_replay_poisson(start_time, end_time, model_mapping_strategy, arrival_distribution, interval_seconds):
    azure_v2_trace.replay(models,
                          model_mapping_strategy=model_mapping_strategy,
                          start_time=start_time,
                          end_time=end_time,
                          arrival_distribution=arrival_distribution,
                          interval_seconds=interval_seconds)


# azure_v1_trace.replay(models, model_mapping_strategy="round_robin",
#                       start_time="0.0.0",
#                       end_time="1.0.0",
#                       arrival_distribution="exponential",
#                       interval_seconds=3600)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", "-s", __file__]))
