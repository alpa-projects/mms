import argparse
import asyncio
import concurrent.futures
from concurrent.futures import wait
import time

import requests

from alpa.util import to_str_round
from alpa_serve.simulator.workload import Workload


class Client:
    def __init__(self, url, max_workers=20):
        self.url = url
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        self.futures = dict()
        self.res_dict = dict()

    @staticmethod
    def submit_one(url, model_name, start, idx):
        while time.time() < start:
            pass

        json = {
            "input": f"I like this movie {idx}",
        }
        res = requests.post(url=url, params={"model": model_name}, json=json)
        assert res.status_code == 200, f"{res.json()}"
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
        self.futures[workload] = futures

    async def wait_all(self):
        for futures in self.futures.values():
            wait(futures)

        for workload, futures in self.futures.items():
            res = [f.result() for f in futures]
            start, finish = zip(*res)
            self.res_dict[workload] = (start, finish)

        self.futuress = dict()

    def print_stats(self, workload: Workload, warmup: float):
        start, finish = self.res_dict[workload]
        workload.print_stats(start, finish, warmup)


def generate_workload(workload, start=0):
    if workload == "tmp":
        w1 = Workload.gen_poisson("a", start, 8, 60, seed=1)
        w2 = Workload.gen_poisson("b", start, 8, 60, seed=2)
        w = w1 + w2
    else:
        raise ValueError(f"Invalid workload name: {workload}")

    return w


async def run_workload(client, workload):
    client.submit_workload(workload)

    await client.wait_all()

    client.print_stats(workload, warmup=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="tmp")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"

    # Launch the client
    client = Client(url)
    workload = generate_workload(args.workload, start=time.time() + 2)

    # Run workloads
    asyncio.run(run_workload(client, workload))
