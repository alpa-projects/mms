import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from alpasim.cluster import ScheduledTask

def compute_statistics_from_cluster_trace(trace_file: str):
    with open(trace_file, "r") as f:
        traces = json.load(f)
    latencies = dict.fromkeys(traces.keys())
    for model_id, trace in traces.items():
        latencies[model_id] = []
        for a, (_, e, _, _) in zip(trace["arrive"], trace["stage_exec_info"][-1]):
            latencies[model_id].append(e - a)
        latencies[model_id] = np.array(latencies[model_id])
    overall_latencies = np.concatenate(list(latencies.values()), axis=None)
    print("-------------------------------")
    print(f"{trace_file} statistics:")
    print(f"overall mean latency: {np.mean(overall_latencies):.3f}s")
    print(f"overall 99% tail latency: {np.quantile(overall_latencies, 0.99):.3f}s")
    for model_id, l in latencies.items():
        print(f"model{model_id} mean latency: {np.mean(l):.3f}s")
        print(f"model{model_id} 99% tail latency: {np.quantile(l, 0.99):.3f}s")
    print("-------------------------------")
    return latencies, overall_latencies

def compute_statistics_from_simulation(tasks: List[ScheduledTask]):
    latencies = {}
    for t in tasks:
        _, e, _, _ = t.stage_execution_info[-1][0]
        if t.model_id not in latencies:
            latencies[t.model_id] = []
        latencies[t.model_id].append(e - t.arrive_time)
    overall_latencies = np.concatenate(list(latencies.values()), axis=None)
    print("-------------------------------")
    print(f"simulation statistics:")
    print(f"overall mean latency: {np.mean(overall_latencies):.3f}s")
    print(f"overall 99% tail latency: {np.quantile(overall_latencies, 0.99):.3f}s")
    for model_id, l in latencies.items():
        print(f"model{model_id} mean latency: {np.mean(l):.3f}s")
        print(f"model{model_id} 99% tail latency: {np.quantile(l, 0.99):.3f}s")
    return latencies, overall_latencies

color_list = [
    "thread_state_uninterruptible",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_unknown",
    "background_memory_dump",
    "light_memory_dump",
    "detailed_memory_dump",
    "vsync_highlight_color",
    "generic_work",
    "good",
    "bad",
    "terrible",
    "yellow",
    "olive",
    "rail_response",
    "rail_animation",
    "rail_idle",
    "rail_load",
    "startup",
    "heap_dump_stack_frame",
    "heap_dump_object_type",
    "heap_dump_child_node_arrow",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "cq_build_attempt_runnig",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
]
 
def dump_chrome_tracing_from_simulation(tasks: List[ScheduledTask], filename: str):
    slot_list = []
    def get_color(t):
        return color_list[t.request_id % len(color_list)]

    for t in tasks:
        for stage_num, stage_exec_info in enumerate(t.stage_execution_info):
            for intra_num, (start, end, node_id, gpu_id) in enumerate(stage_exec_info):
                slot = {"name": f"r{t.request_id}.s{stage_num}.{intra_num}",
                        "cat": f"stage{stage_num}, intra{intra_num}",
                        "ph": "X",
                        "pid": node_id,
                        "tid": gpu_id,
                        "ts": start * 1e6,
                        "dur": (end - start) * 1e6,
                        "cname": get_color(t)}
                slot_list.append(slot)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as fout:
        fout.write(json.dumps({
            "traceEvents": slot_list,
            "displayTimeUnit": "ms",
        }))


def dump_chrome_tracing_from_cluster_trace(trace_file: str, dumpfile: str):
    def get_color(i):
        return color_list[i % len(color_list)]
    
    slot_list = []
    with open(trace_file, "r") as f:
        traces = json.load(f)
        for model_id, trace in traces.items():
            for i, stage_exec_info in enumerate(trace["stage_exec_info"]):
                for rq_id, (s, e, node_ids, devices) in zip(trace["rq_id"], stage_exec_info):
                        for node_id, devices_per_node in zip(node_ids, devices):
                            for device in devices_per_node:
                                slot = {"name": f"r{rq_id}s{i}",
                                        "cat": f"model{model_id}",
                                        "ph": "X",
                                        "pid": node_id,
                                        "tid": device,
                                        "ts": float(s) * 1e6,
                                        "dur": float(e - s) * 1e6,
                                        "cname": get_color(rq_id)}
                                slot_list.append(slot)

    os.makedirs(os.path.dirname(dumpfile), exist_ok=True)
    with open(dumpfile, "w") as fout:
        fout.write(json.dumps({
            "traceEvents": slot_list,
            "displayTimeUnit": "ms",
        }))