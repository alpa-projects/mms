"""Workload definition"""
from collections import defaultdict, namedtuple
import dataclasses
import random
from typing import Any, List, Sequence

import numpy as np

from alpa.util import to_str_round


@dataclasses.dataclass
class Request:
    """A single request."""
    model_name: str
    data: Any
    slo: float
    idx: int
    submit_time: float = None   # This will be filled later


StatsResult = namedtuple("StatsResult", ("average_goodput", "per_model_stats"))

PerModelStatsResult = namedtuple("PerModelStatsResult",
        ("name", "num_requests", "goodput", "throughput",
         "latency_mean", "latency_std", "latency_p90", "latency_p99"))


class Workload:
    """A sorted list of requests."""

    def __init__(self, arrivals: List[float], requests: List[Request]):
        assert len(arrivals) == len(requests)

        self.arrivals = arrivals
        self.requests = requests

    def compute_stats(self, start: Sequence[float], finish: Sequence[float],
                      good: Sequence[bool], warmup: float):
        """Compute the statistics of serving results."""
        # Skip the first and last `warmup` seconds
        ct = 1
        while ct < len(start) and start[ct] - start[0] < warmup:
            ct += 1
        start = np.asarray(start[ct:-ct])
        finish = np.asarray(finish[ct:-ct])
        good = np.asarray(good[ct:-ct])
        workload = self[ct:-ct]

        # Compute stats per model
        model_indices = defaultdict(list)
        for i in range(len(workload)):
            model_indices[workload.requests[i].model_name].append(i)

        names = list(model_indices.keys())
        names.sort()

        stats = []
        goodputs = []
        for name in names:
            indices = model_indices[name]
            tmp_good = good[indices]
            tmp_start = start[indices][tmp_good]
            tmp_finish = finish[indices][tmp_good]

            # Compute stats
            goodput = np.sum(tmp_good) / len(tmp_good)
            if goodput > 0:
                throughput = len(tmp_start) / (tmp_finish[-1] - tmp_start[0])
                latency = tmp_finish - tmp_start
            else:
                throughput = 0
                latency = [0]

            sorted_latency = np.sort(latency)
            latency_p90 = sorted_latency[int(0.90 * len(sorted_latency))]
            latency_p99 = sorted_latency[int(0.99 * len(sorted_latency))]

            stats.append(PerModelStatsResult(
                name, len(indices), goodput, throughput,
                np.mean(latency), np.std(latency),
                latency_p90, latency_p99))
            goodputs.append(goodput)

        return StatsResult(np.mean(goodputs), stats)

    @staticmethod
    def print_stats(stats: StatsResult):
        """Print the statistics of serving results."""
        for stat in stats.per_model_stats:
            print(f"model: {stat.name}, #req: {stat.num_requests}")
            print(f"goodput: {stat.goodput*100:.2f} %")
            print(f"throughput: {stat.throughput:.2f} q/s")
            print(f"latency mean: {stat.latency_mean*1e3:.2f} ms, "
                  f"std: {stat.latency_std*1e3:.2f} ms, "
                  f"p90: {stat.latency_p90*1e3:.2f} ms")
        print(f"average goodput: {stats.average_goodput*100:.2f} %")

    @staticmethod
    def gen_uniform(model_name: str, start: float, throughput: float,
                    duration: float, slo: float=1, seed: int=0):
        number = int(duration * throughput)
        interval = 1 / throughput
        ticks = [start + i * interval for i in range(number)]
        return Workload(ticks, [
            Request(model_name, None, slo, i) for i in range(number)])

    @staticmethod
    def gen_poisson(model_name: str, start: float, throughput: float,
                    duration: float, slo: float=1, seed: int=0):
        random.seed(seed)

        number = int(duration * throughput)
        ticks = []
        cur = start
        for i in range(number):
            cur += random.expovariate(throughput)
            ticks.append(cur)
        return Workload(ticks, [
            Request(model_name, None, slo, i) for i in range(number)])

    @staticmethod
    def empty():
        return Workload([], [])

    @staticmethod
    def merge(*args):
        if len(args) == 1:
            return args[0]

        number = sum(len(x) for x in args)

        merged_arrivals = sum((x.arrivals for x in args), [])
        merged_requests = sum((x.requests for x in args), [])

        sorted_indices = np.argsort(merged_arrivals)

        arrivals = [None] * number
        requests = [None] * number

        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]

        return Workload(arrivals, requests)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.arrivals)))
            arrivals = [self.arrivals[i] for i in indices]
            requests = [self.requests[i] for i in indices]
            return Workload(arrivals, requests)
        else:
            raise NotImplementedError

    def __add__(self, other):
        return Workload.merge(self, other)

    def __len__(self):
        return len(self.arrivals)

    def __str__(self):
        return (f"Workload(len={len(self)}, "
                f"arrivals={to_str_round(self.arrivals[:20])}...)")


if __name__ == "__main__":
    a = Workload.gen_poisson(0, 0, 10, 60)
    b = Workload.gen_uniform(1, 0, 10, 60)
    c = a + b

    print(c)
