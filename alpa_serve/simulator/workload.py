import dataclasses
import random
from typing import Any, List

import numpy as np
from alpa.util import to_str_round


@dataclasses.dataclass
class Request:
    name: str
    data: Any


class Workload:
    def __init__(self, arrivals: List[float], requests: List[Request]):
        assert len(arrivals) == len(requests)

        self.arrivals = arrivals
        self.requests = requests

    @staticmethod
    def gen_uniform(name: str, start: float, throughput: float,
                    duration: float, seed: int=0):
        number = int(duration * throughput)
        interval = 1 / throughput
        ticks = [start + i * interval for i in range(number)]
        return Workload(ticks, [Request(name, None)] * number)

    @staticmethod
    def gen_poisson(name: str, start: float, throughput: float,
                    duration: float, seed: int=0):
        random.seed(seed)

        number = int(duration * throughput)
        ticks = []
        cur = start
        for i in range(number):
            cur += random.expovariate(throughput)
            ticks.append(cur)
        return Workload(ticks, [Request(name, None)] * number)

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

    def __len__(self):
        return len(self.arrivals)

    def __str__(self):
        return (f"Workload(len={len(self)}, "
                f"arrivals={to_str_round(self.arrivals[:20])}...)")


if __name__ == "__main__":
    a = Workload.gen_poisson(0, 0, 10, 60)
    b = Workload.gen_uniform(1, 0, 10, 60)
    c = Workload.merge(a, b)

    print(c)
