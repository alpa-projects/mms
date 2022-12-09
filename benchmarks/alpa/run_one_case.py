import argparse
import asyncio
import concurrent.futures
from concurrent.futures import wait
from multiprocessing import get_context
import time

import requests
import ray
import numpy as np

from alpa_serve.controller import run_controller, CONTROLLER_NAME
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.simulator.workload import Workload, DEFAULT_WARMUP
from alpa_serve.simulator.controller import run_workload
from alpa_serve.util import ServingCase, to_str_round


global global_controller

def worker_initializer(url):
    global global_controller

    if url is None:  # url is None means using ray
        ray.init(address="auto", namespace="alpa_serve")
        global_controller = ray.get_actor(CONTROLLER_NAME)


def submit_one(arg):
    url, model_name, slo, start, idx, relax_slo, debug = arg
    if time.time() > start:
        pass #print(f"WARNING: Request {idx} is blocked by the client. ")

    while time.time() < start:
        pass

    obj = {
        "model": model_name,
        "submit_time": start,
        "slo": slo,
        "idx": idx,
        "input": f"I like this movie {idx}",
    }

    if url is None:
        res = ray.get(global_controller.handle_request.remote(obj))
        status_code = 200
    else:
        res = requests.post(url=url, json=obj)
        status_code, res = res.status_code, res.json()

    assert status_code == 200, f"{res}"
    end = time.time()
    e2e_latency = end - start
    rejected = res["rejected"]
    good = e2e_latency <= slo and not rejected
    if relax_slo:
        good = not rejected

    if e2e_latency > slo and not rejected:
        print(f"WARNING: Request {idx} is accepted but not good. ")

    if debug:
        tstamps = to_str_round({x: (y - start) * 1e3 for x, y in res["ts"]}, 2)
        print(f"idx: {idx} ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms")

    return start, end, good


class ProcessPoolClient:
    def __init__(self, url, relax_slo=False, debug=False, max_workers=20):
        self.url = url
        self.relax_slo = relax_slo
        self.debug = debug
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=get_context("spawn"),
            initializer=worker_initializer, initargs=(url,))
        self.res_dict = dict()

    async def submit_workload(self, workload: Workload):
        args = [(self.url, workload.requests[i].model_name,
                 workload.requests[i].slo, float(workload.arrivals[i]),
                 i, self.relax_slo, self.debug) for i in range(len((workload)))]
        res = self.executor.map(submit_one, args)

        start, finish, good = zip(*res)
        self.res_dict[workload] = (
            np.asarray(start, dtype=np.float64),
            np.asarray(finish, dtype=np.float64),
            np.asarray(good, dtype=bool))

    def compute_stats(self, workload: Workload, warmup: float):
        start, finish, good = self.res_dict[workload]
        return workload.compute_stats(start, finish, good, warmup)

    def __del__(self):
        self.executor.shutdown()
        self.executor = self.res_dict = None


def run_one_case(case: ServingCase, warmup=DEFAULT_WARMUP,
                 relax_slo=False, debug=False,
                 protocol="http", port=20001):
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = run_controller("localhost", port=port)
    register_models(controller)
    placement = place_models(controller)
    controller.warmup.remote()
    controller.sync()

    # Launch the client
    url = f"http://localhost:{port}" if protocol == "http" else None
    client = ProcessPoolClient(url, relax_slo, debug)
    workload = generate_workload(start=time.time() + 2)

    # Run workloads
    stats = asyncio.run(run_workload(client, workload, warmup))
    ray.get(controller.shutdown.remote())
    del controller, client
    return stats, placement


if __name__ == "__main__":
    from benchmarks.alpa.suite_debug import suite_debug

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_replicate")
    parser.add_argument("--relax-slo", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--protocol", choices=["http", "ray"], default="http")
    args = parser.parse_args()

    ray.init(address="auto", namespace="alpa_serve")

    stats, placement = run_one_case(
        suite_debug[args.case], relax_slo=args.relax_slo, debug=args.debug,
        protocol=args.protocol)
    Workload.print_stats(stats)
