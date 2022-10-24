import argparse
import concurrent.futures
from concurrent.futures import wait
import logging
import time

import numpy as np
import ray
import requests

from alpa.util import to_str_round
from alpa_serve.simulator.workload import Workload
from benchmarks.alpa.util import build_logger


class Client:
    def __init__(self, url, max_workers=20):
        self.url = url
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    @staticmethod
    def submit_one(url, model_name, start, idx):
        while time.time() < start:
            pass

        json = {
            "input": f"I like this movie {idx}",
        }
        res = requests.post(url=url, params={"model": model_name}, json=json)
        assert res.status_code == 200
        end = time.time()

        res = res.json()
        e2e_latency = end - start
        tstamps = to_str_round({x: (y - start) * 1e3 for x, y in res["ts"]}, 2)
        print(f"ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms", flush=True)
        return start, end

    def submit_workload(self, workload: Workload):
        futures = [self.executor.submit(Client.submit_one, url,
                                        workload.requests[i].model_name,
                                        workload.arrivals[i], i)
                   for i in range(len((workload)))]
        return futures
 
    def print_stats(self, workload: Workload, futures, warmup):
        if not futures:
            return
        res = [f.result() for f in futures]
        start, finish = zip(*res)

        workload.print_stats(start, finish, warmup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"

    client = Client(url)

    tic = time.time() + 2
    w1 = Workload.gen_poisson("alpa/bert-1", tic, 6, 60, seed=1)
    w2 = Workload.gen_poisson("alpa/bert-2", tic, 6, 60, seed=2)
    w = w1 + w2

    fs = client.submit_workload(w)
    assert tic > time.time()

    wait(fs)
    client.print_stats(w, fs, warmup=10)
