import argparse
import concurrent.futures
from concurrent.futures import wait
import logging
import random
import time

import numpy as np
import ray
import requests

from alpa.util import to_str_round
from benchmarks.alpa.util import build_logger


class Client:
    def __init__(self, url, model_name):
        self.model_name = model_name
        self.url = url
    
    def query(self, inputs):
        json = {
            "input": inputs,
        }
        resp = requests.post(
            url=self.url, params={"model": self.model_name}, json=json)
        return resp


def submit_request(client, start, idx):
    while time.time() < start:
        pass

    res = client.query(f"I like this movie {idx}")
    assert res.status_code == 200
    end = time.time()

    res = res.json()
    e2e_latency = end - start
    tstamps = to_str_round({x: (y - start) * 1e3 for x, y in res["ts"]}, 2)
    print(f"ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms", flush=True)
    return start, end


class RequestSubmitter:
    def __init__(self, url, model_name, max_workers=10):
        self.client = Client(url, model_name)
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    def warmup(self):
        futures = self.submit_uniform(time.time(), 1, 2)

    def submit_uniform(self, start, throughput, duration):
        number = int(duration * throughput)
        interval = 1 / throughput
        ticks = [start + i * interval for i in range(number)]
        return self.submit_request_internal(ticks)

    def submit_poisson(self, start, throughput, duration):
        number = int(duration * throughput)
        ticks = []
        cur = start
        for i in range(number):
            cur += random.expovariate(throughput)
            ticks.append(cur)
        return self.submit_request_internal(ticks)

    def submit_request_internal(self, request_start):
        futures = [self.executor.submit(submit_request, self.client, s, i)
                   for i, s in enumerate(request_start)]
        return futures
 
    def print_stats(self, futures, warmup=10):
        if not futures:
            return
        res = [f.result() for f in futures]
        request_start, request_end = zip(*res)

        # Skip the first and last `warmup` seconds
        ct = 0
        while request_start[ct] - request_start[0] < warmup:
            ct += 1
        request_start = np.array(request_start[ct:-ct])
        request_end = np.array(request_end[ct:-ct])

        # Compute stats
        throughput = len(request_start) / (request_end[-1] - request_start[0])
        latency = request_end - request_start
        sorted_latency = np.sort(latency)
        average_latency = np.mean(latency)
        tail_latnecy_90 = sorted_latency[int(0.90 * len(sorted_latency))]
        tail_latnecy_99 = sorted_latency[int(0.99 * len(sorted_latency))]

        print(f"#req: {len(latency)}")
        print(f"throughput: {throughput:.2f} q/s")
        print(f"latency mean: {np.mean(latency)*1e3:.2f} ms, "
              f"std: {np.std(latency)*1e3:.2f} ms, "
              f"p90 : {tail_latnecy_90*1e3:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    url = f"http://localhost:{args.port}"

    s1 = RequestSubmitter(url, "alpa/bert-1")
    s2 = RequestSubmitter(url, "alpa/bert-2")

    s1.warmup()
    s2.warmup()

    tic = time.time() + 2
    res1 = s1.submit_uniform(tic, 12, 60)
    res2 = s2.submit_poisson(tic,  2, 60)
    assert tic > time.time()

    wait(res1 + res2)

    s1.print_stats(res1)
    s2.print_stats(res2)
