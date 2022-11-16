import argparse
import asyncio
import concurrent.futures
from concurrent.futures import wait
import time

import requests
import ray

from alpa_serve.controller import run_controller
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.simulator.workload import Workload
from alpa_serve.util import ServingCase
from alpa.util import to_str_round

from benchmarks.alpa.suite_debug import suite_debug


class Client:
    def __init__(self, url, max_workers=10):
        self.url = url
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        self.futures = dict()
        self.res_dict = dict()

    @staticmethod
    def submit_one(url, model_name, slo, start, idx):
        while time.time() < start:
            time.sleep(0.0001)

        json = {
            "model": model_name,
            "submit_time": start,
            "slo": slo,
            "input": f"I like this movie {idx}",
        }
        res = requests.post(url=url, json=json)
        status_code, res = res.status_code, res.json()
        assert status_code == 200, f"{res}"
        end = time.time()
        e2e_latency = end - start
        good = e2e_latency <= slo and status_code == 200 and not res["rejected"]

        tstamps = to_str_round({x: (y - start) * 1e3 for x, y in res["ts"]}, 2)
        print(f"idx: {idx} ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms", flush=True)
        return start, end, good

    def submit_workload(self, workload: Workload):
        futures = [self.executor.submit(Client.submit_one, self.url,
                                        workload.requests[i].model_name,
                                        workload.requests[i].slo,
                                        workload.arrivals[i], i)
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


async def run_workload(client, workload):
    client.submit_workload(workload)

    await client.wait_all()

    return client.compute_stats(workload, warmup=10)


def run_one_case(case: ServingCase, port=20001):
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = run_controller("localhost", port=port, name=None)
    register_models(controller)
    placement_policy = place_models(controller)

    # Launch the client
    client = Client(f"http://localhost:{port}")
    workload = generate_workload(start=time.time() + 2)

    # Run workloads
    stats = asyncio.run(run_workload(client, workload))
    ray.get(controller.shutdown.remote())
    del controller, client
    return stats, placement_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_manual_1")
    args = parser.parse_args()

    ray.init(address="auto")

    stats, _ = run_one_case(suite_debug[args.case])
    Workload.print_stats(stats)
