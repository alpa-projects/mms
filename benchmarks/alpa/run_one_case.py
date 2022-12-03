import argparse
import asyncio
import concurrent.futures
from concurrent.futures import wait
import time

import requests
import ray

from alpa_serve.controller import run_controller
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.simulator.workload import Workload, DEFAULT_WARMUP
from alpa_serve.simulator.controller import run_workload
from alpa_serve.util import ServingCase
from alpa.util import to_str_round

from benchmarks.alpa.suite_debug import suite_debug


class Client:
    def __init__(self, url, debug=False, max_workers=25):
        self.url = url
        self.debug = debug
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        self.futures = dict()
        self.res_dict = dict()

    @staticmethod
    def submit_one(url, model_name, slo, start, idx, debug):
        if time.time() > start:
            print(f"WARNING: Request {idx} is blocked by the client.")

        while time.time() < start:
            pass

        json = {
            "model": model_name,
            "submit_time": start,
            "slo": slo,
            "idx": idx,
            "input": f"I like this movie {idx}",
        }
        res = requests.post(url=url, json=json)
        status_code, res = res.status_code, res.json()
        assert status_code == 200, f"{res}"
        end = time.time()
        e2e_latency = end - start
        good = e2e_latency <= slo and not res["rejected"]

        if e2e_latency > slo and not res["rejected"]:
            print(f"WARNING: Request {idx} is accepted but not good.")

        if debug:
            tstamps = to_str_round({x: (y - start) * 1e3 for x, y in res["ts"]}, 2)
            print(f"idx: {idx} ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms", flush=True)

        return start, end, good

    def submit_workload(self, workload: Workload):
        futures = [self.executor.submit(Client.submit_one, self.url,
                                        workload.requests[i].model_name,
                                        workload.requests[i].slo,
                                        workload.arrivals[i], i,
                                        self.debug)
                   for i in range(len((workload)))]
        self.futures[workload] = futures

    async def wait_all(self):
        for futures in self.futures.values():
            wait(futures)

        for workload, futures in self.futures.items():
            res = [f.result() for f in futures]
            start, finish, good = zip(*res)
            self.res_dict[workload] = (start, finish, good)

        self.futuress = dict()

    def compute_stats(self, workload: Workload, warmup: float):
        start, finish, good = self.res_dict[workload]
        return workload.compute_stats(start, finish, good, warmup)

    def __del__(self):
        self.futures = self.executor = self.res_dict = None


def run_one_case(case: ServingCase, warmup=DEFAULT_WARMUP, debug=False, port=20001):
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = run_controller("localhost", port=port, name=None)
    register_models(controller)
    placement = place_models(controller)

    # Launch the client
    client = Client(f"http://localhost:{port}", debug)
    workload = generate_workload(start=time.time() + 2)

    # Run workloads
    stats = asyncio.run(run_workload(client, workload, warmup))
    ray.get(controller.shutdown.remote())
    del controller, client
    return stats, placement


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_replicate")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ray.init(address="auto")

    stats, placement = run_one_case(suite_debug[args.case], debug=args.debug)
    Workload.print_stats(stats)
