import argparse
import asyncio
import concurrent.futures
from concurrent.futures import wait
import time

import requests
import ray

from alpa_serve import run_controller
from alpa_serve.simulator.workload import Workload
from alpa.util import to_str_round

from benchmarks.alpa.suite import cases


class Client:
    def __init__(self, url, max_workers=10, debug=False):
        self.url = url
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self.debug = debug

        self.futures = dict()
        self.res_dict = dict()

    def submit_one(self, url, model_name, slo, start, idx):
        while time.time() < start:
            pass

        json = {
            "input": f"I like this movie {idx}",
        }
        res = requests.post(url=url, params={"model": model_name}, json=json)
        assert res.status_code == 200 or res.status_code == 503, f"{res.json()}"
        end = time.time()
        e2e_latency = end - start
        good = e2e_latency <= slo and res.status_code == 200

        res = res.json()
        if self.debug:
            tstamps = to_str_round({x: (y - start) * 1e3 for x, y in res["ts"]}, 2)
            print(f"idx: {idx} ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms", flush=True)
        return start, end, good

    def submit_workload(self, workload: Workload):
        futures = [self.executor.submit(self.submit_one, self.url,
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


def run_one_case(case, port=20001, debug=False):
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = run_controller("localhost", port=port, name=None)
    register_models(controller)
    place_models(controller)

    # Launch the client
    client = Client(f"http://localhost:{port}", debug=debug)
    workload = generate_workload(start=time.time() + 2)

    # Run workloads
    return asyncio.run(run_workload(client, workload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_manual_1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ray.init(address="auto")

    stats = run_one_case(cases[args.case], debug=args.debug)
    Workload.print_stats(stats)
