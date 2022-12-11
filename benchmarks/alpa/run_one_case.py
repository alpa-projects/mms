import argparse
import asyncio
import concurrent.futures
from concurrent.futures import wait
from multiprocessing import get_context, Process
import threading
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
        pass #print(f"WARNING: Request {idx} is blocked by the client. "
              #f"{time.time() - start:.4f}")

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
        if relax_slo:
            print(f"WARNING: Request {idx} is accepted but not good. (relaxed)")
        else:
            print(f"WARNING: Request {idx} is accepted but not good.")

    if debug:
        tstamps = to_str_round({x: (y - start) * 1e3 for x, y in res["ts"]}, 2)
        print(f"idx: {idx} ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms")

    return start, end, good


class ProcessPoolClient:
    def __init__(self, url, relax_slo=False, debug=False, max_workers=30):
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


class ThreadClient:
    def __init__(self, url, relax_slo=False, debug=False):
        self.url = url
        self.relax_slo = relax_slo
        self.debug = debug
        self.res_dict = dict()

        # warmup
        asyncio.run(self.submit_workload(Workload.empty()))

    @staticmethod
    def worker_func(arg, i, start, finish, good):
        s, f, g = submit_one(arg)
        start[i], finish[i], good[i] = s, f, g

    async def submit_workload(self, workload: Workload):
        args = [(self.url, workload.requests[i].model_name,
                 workload.requests[i].slo, float(workload.arrivals[i]),
                 i, self.relax_slo, self.debug) for i in range(len((workload)))]
        num_requests = len(workload)
        start = np.zeros(num_requests, dtype=np.float64)
        finish = np.zeros(num_requests, dtype=np.float64)
        good = np.zeros(num_requests, dtype=bool)

        ts = [None] * int(workload.rate * 10)

        for i in range(num_requests):
            while time.time() < workload.arrivals[i] - 0.010:
                pass

            t = threading.Thread(
                    target=ThreadClient.worker_func,
                    args=(args[i], i, start, finish, good))
            t.start()

            ts[i % len(ts)] = t

        for t in ts:
            if t:
                t.join()

        self.res_dict[workload] = (start, finish, good)

    def compute_stats(self, workload: Workload, warmup: float):
        start, finish, good = self.res_dict[workload]
        return workload.compute_stats(start, finish, good, warmup)

    def __del__(self):
        self.res_dict = None


class ParallelThreadClient:
    def __init__(self, url, relax_slo=False, debug=False, max_workers=10):
        self.url = url
        self.relax_slo = relax_slo
        self.debug = debug
        self.max_workers = max_workers
        self.res_dict = dict()
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=get_context("spawn"),
            initializer=worker_initializer, initargs=(url,))

        # warmup
        asyncio.run(self.submit_workload(Workload.empty()))

    @staticmethod
    def worker_func(arg):
        url, relax_slo, debug, workload = arg
        client = ThreadClient(url, relax_slo=relax_slo, debug=debug)
        asyncio.run(client.submit_workload(workload))
        return client.res_dict[workload]

    async def submit_workload(self, workload: Workload):
        ws = workload.split_round_robin(self.max_workers)
        args = [(self.url, self.relax_slo, self.debug, w) for w in ws]
        results = self.executor.map(ParallelThreadClient.worker_func, args)

        num_requests = len(workload)
        start = np.zeros(num_requests, dtype=np.float64)
        finish = np.zeros(num_requests, dtype=np.float64)
        good = np.zeros(num_requests, dtype=bool)
        for i, res in enumerate(results):
            pt = i
            for s, f, g in zip(*res):
                start[pt], finish[pt], good[pt] = s, f, g
                pt += self.max_workers

        self.res_dict[workload] = (start, finish, good)

    def compute_stats(self, workload: Workload, warmup: float):
        start, finish, good = self.res_dict[workload]
        return workload.compute_stats(start, finish, good, warmup)

    def __del__(self):
        self.res_dict = None


def run_one_case(case: ServingCase, warmup=DEFAULT_WARMUP,
                 relax_slo=False, debug=False,
                 protocol="http", port=20001):
    register_models, generate_workload, place_models = case

    # Launch the controller
    if not ray.is_initialized():
        ray.init(address="auto", namespace="alpa_serve")
    controller = run_controller("localhost", port=port)
    register_models(controller)
    placement = place_models(controller)
    controller.warmup.remote()
    controller.sync()

    # Launch the client
    workload = generate_workload(start=time.time() + 5)
    url = f"http://localhost:{port}" if protocol == "http" else None
    slo = np.mean([r.slo for r in workload.requests[0:10]])
    if slo < 0.4:
        client = ProcessPoolClient(url, relax_slo, debug)
    else:
        client = ThreadClient(url, relax_slo, debug)

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

    stats, placement = run_one_case(
        suite_debug[args.case], relax_slo=args.relax_slo, debug=args.debug,
        protocol=args.protocol)
    Workload.print_stats(stats)
