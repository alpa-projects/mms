"""Workload definition"""
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import dataclasses
import random
from typing import Any, List, Sequence, Dict, Optional

import numpy as np

from alpa_serve.simulator.util import MMPPSampler
from alpa_serve.util import to_str_round, eps


DEFAULT_WARMUP = 10


@dataclasses.dataclass
class Request:
    """A single request."""
    model_name: str
    data: Any
    slo: Optional[float]
    idx: int
    time_stamp: Dict            # debug only
    submit_time: float = None   # This will be filled later


PerModelStatsResult = namedtuple("PerModelStatsResult",
        ("name", "num_requests", "goodput", "throughput",
         "latency_mean", "latency_std", "latency_p90", "latency_p99", "latency",
         "request_starts", "request_finishes"))

PerDeviceStatsResult = namedtuple("PerDeviceStatsResult", ("num_requests",))


@dataclasses.dataclass
class StatsResult:
    per_model_stats: List[PerModelStatsResult]
    group_num_requests: List[int]
    goodput: float
    latency_mean: float
    num_requests: int
    request_rate: float


class ArrivalProcess(ABC):
    @abstractmethod
    def rate(self):
        """Return the mean arrival rate."""
        raise NotImplementedError()

    @abstractmethod
    def cv(self):
        """Return the coefficient of variation of the gap between
        the requests."""
        raise NotImplementedError()

    @abstractmethod
    def generate_arrivals(self, start: float, duration: float,
                          seed: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def generate_workload(self, model_name: str, start: float,
                          duration: float, slo: Optional[float] = None,
                          seed: int = 0):
        """Generate a workload with the arrival process.

        Args:
            model_name (str): Name of the model.
            start (float): The start time of the workload.
            duration (float): The duration of the workload.
            slo (Optional[float]): The service level objective of each model.
            seed (int): The random seed.
        """
        raise NotImplementedError()

    def __str__(self):
        return (f"{self.__class__.__name__}("
                f"rate={self.rate()}, "
                f"cv={self.cv()})")

    def params(self):
        return self.rate(), self.cv()


class DeterministicProcess(ArrivalProcess):
    """Deterministic arrival process."""
    def __init__(self, arrival_rate: float):
        """Create a deterministic arrival process.

        Args:
            arrival_rate (float): The arrival rate of the process. The gap
                between the requests is 1 / arrival_rate seconds.
        """
        self.rate_ = arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return 0

    def generate_workload(self, model_name: str, start: float,
                          duration: float, slo: Optional[float] = None,
                          seed: int = 0):
        n_requests = int(duration * self.rate_)
        interval = 1 / self.rate_
        ticks = [start + i * interval for i in range(n_requests)]
        return Workload(ticks, [
            Request(model_name, None, slo, i, {}) for i in range(n_requests)])


class GammaProcess(ArrivalProcess):
    """Gamma arrival process."""
    def __init__(self, arrival_rate: float, cv: float):
        """Initialize a gamma arrival process.

        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate
        self.cv_ = cv
        self.shape = 1 / (cv * cv)
        self.scale = cv * cv / arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return self.cv_

    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        np.random.seed(seed)

        batch_size = max(int(self.rate_ * duration * 1.2), 1)
        intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
        pt = 0

        ticks = []
        cur = start + intervals[0]
        end = start + duration
        while cur < end:
            ticks.append(cur)

            pt += 1
            if pt >= batch_size:
                intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
                pt = 0

            cur += intervals[pt]

        return ticks

    def generate_workload(self, model_name: str, start: float,
                          duration: float, slo: Optional[float] = None,
                          seed: int = 0):
        ticks = self.generate_arrivals(start, duration, seed)
        return Workload(ticks, [
            Request(model_name, None, slo, i, {}) for i in range(len(ticks))])


class PoissonProcess(GammaProcess):
    """Poisson arrival process."""

    def __init__(self, arrival_rate: float):
        """Initialize a Poisson arrival process.

        Args:
            arrival_rate: The mean arrival rate.
        """
        super().__init__(arrival_rate, 1)


class UniformMMPP(ArrivalProcess):
    """Markov Modulated Poisson Process (MMPP), where the transition
    probability among the states of the Markov chain is uniform
    across all states.

    MMPP is a generalization of the Poisson process where the request rate
    changes over time. An m-state MMPP can be viewed as m independent
    Poisson processes with different request rates. A switch governed by
    an m-state Markov chain determines which of m request processes is
    active, i.e., the one in accordance with which requests are generated.
    The duration staying on each state is exponentially distributed with
    the provided mean duration of each state. In this simplified unifrom
    case, we assume the transition probability among the states of the
    Markov chain is uniform across all states (i.e., each state will
    transit to another state with equal probability across all other
    states).
    """
    def __init__(self, state_durations: Sequence[float],
                 state_request_rates: Sequence[float]):
        """Initialize a uniform MMPP.

        Args:
            state_durations: The duration of each state.
            state_request_rates: The request rate of each state.
        """
        self.state_durations = np.array(state_durations)
        self.state_request_rates = np.array(state_request_rates)
        assert len(self.state_durations) == len(self.state_request_rates)
        self.mean_arrival_rate = (np.sum(self.state_durations
                                         * self.state_request_rates)
                                  / np.sum(self.state_durations))

    def rate(self):
        return self.mean_arrival_rate

    def cv(self):
        return None

    def generate_workload(self, model_name: str, start: float,
                          duration: float, slo: Optional[float] = None,
                          seed: int = 0):
        np.random.seed(seed)
        random.seed(seed)
        n_requests = int(duration * self.mean_arrival_rate)
        sampler = MMPPSampler.unifrom_mmpp(self.state_durations,
                                           self.state_request_rates)
        ticks, _ = sampler.sample(n_requests)
        ticks = [start + t for t in ticks[1:]]
        return Workload(ticks, [
            Request(model_name, None, slo, i, {}) for i in range(n_requests)])


class ParetoProcess:
    def __init__(self, shape, scale, loc = 0.0):
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        from scipy.stats import pareto

        rs = np.random.RandomState(seed)
        ticks = []
        cur = start
        end = start + duration
        while cur < end:
            cur += pareto.rvs(self.shape, loc = self.loc, scale = self.scale, random_state=rs) - 1.0
            ticks.append(cur)
        return ticks

    def generate_workload(self, model_name: str, start: float,
                          duration: float, slo: Optional[float] = None,
                          seed: int = 0):
        ticks = self.generate_arrivals(start, duration, seed)
        return Workload(ticks, [
            Request(model_name, None, slo, i, {}) for i in range(len(ticks))])

    def rate(self):
        """TODO(Hao): this is wrong."""
        return 1.0

    def cv(self):
        """TODO(Hao): this is wrong."""
        return 1.0

    def params(self):
        return self.rate(), self.cv()


class Workload:
    """A sorted list of requests."""

    def __init__(self, arrivals: List[float], requests: List[Request]):
        assert len(arrivals) == len(requests)

        self.arrivals = np.array(arrivals)
        self.requests = requests

        self.enable_simulator_cache = False
        self.cached_data = None

        if len(self.arrivals) > 1:
            intervals = self.arrivals[1:] - self.arrivals[:-1]
            self.rate = 1 / (np.mean(intervals) + eps)
            self.cv = np.std(intervals) * self.rate
        else:
            self.rate = 0
            self.cv = 0

    def split_round_robin(self, number: int):
        rets = []
        for i in range(number):
            rets.append(self[i::number])
        return rets

    def split_time_interval(self, interval: float):
        if len(self.arrivals) < 1:
            return []

        ws = []
        start_i = 0
        start_time = self.arrivals[start_i]
        for i in range(len(self.arrivals)):
            if self.arrivals[i] > start_time + interval:
                ws.append(self[start_i:i])
                start_i = i
                start_time = self.arrivals[i]

        ws.append(self[start_i:])
        return ws

    def compute_stats(self, start: Sequence[float], finish: Sequence[float],
                      good: Sequence[bool], warmup: float):
        """Compute the statistics of serving results."""
        # Skip the first and last `warmup` seconds
        if len(self.arrivals) > 1:
            skip = int(warmup / (self.arrivals[-1] - self.arrivals[0]) * len(self.arrivals))
            if skip > 0:
                start = start[skip:-skip]
                finish = finish[skip:-skip]
                good = good[skip:-skip]
                requests = self.requests[skip:-skip]

        # Compute stats per model
        model_indices = defaultdict(list)
        for i in range(len(requests)):
            model_indices[requests[i].model_name].append(i)

        names = list(model_indices.keys())
        names.sort(key=lambda name: len(model_indices[name]))

        stats = []
        for name in names:
            indices = np.asarray(model_indices[name], dtype=np.int32)
            tmp_good = np.asarray(good[indices], dtype=bool)
            tmp_start = start[indices][tmp_good]
            tmp_finish = finish[indices][tmp_good]

            # Compute stats
            goodput = np.mean(tmp_good)
            if goodput > 0:
                throughput = len(tmp_start) / (tmp_start[-1] - tmp_start[0])
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
                latency_p90, latency_p99, latency, tmp_start, tmp_finish))

        return StatsResult(stats, None, np.mean(good), np.mean(finish - start),
                           len(start), len(start) / (start[-1] - start[0]))

    @staticmethod
    def print_stats(stats: StatsResult):
        """Print the statistics of serving results."""
        if stats.per_model_stats:
            print("--- per model ---")
            for stat in stats.per_model_stats:
                print(f"model: {stat.name}, #req: {stat.num_requests}")
                print(f"goodput: {stat.goodput*100:.2f} %, "
                      f"throughput: {stat.throughput:.2f} q/s, ")
                      #f"#bad: {int(stat.num_requests * (1-stat.goodput))}")
                print(f"latency mean: {stat.latency_mean*1e3:.2f} ms, "
                      f"std: {stat.latency_std*1e3:.2f} ms, "
                      f"p90: {stat.latency_p90*1e3:.2f} ms")
        if stats.group_num_requests is not None:
            print(f"per group #req: {stats.group_num_requests}")
        print("--- overall ---")
        print(f"total #req: {stats.num_requests}, "
              f"rate: {stats.request_rate:.2f} q/s")
        print(f"average goodput: {stats.goodput*100:.2f} %, "
              f"latency mean: {stats.latency_mean*1e3:.2f} ms")

    @classmethod
    def empty(cls):
        return cls([], [])

    @classmethod
    def merge(cls, *args):
        if len(args) == 1:
            return args[0]

        number = sum(len(x) for x in args)

        merged_arrivals = np.concatenate(tuple(x.arrivals for x in args))
        merged_requests = sum((x.requests for x in args), [])

        sorted_indices = np.argsort(merged_arrivals)

        arrivals = [None] * number
        requests = [None] * number

        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]
            requests[i].idx = i

        return cls(arrivals, requests)

    def __getitem__(self, key):
        if isinstance(key, slice):
            arrivals = self.arrivals.__getitem__(key)
            requests = self.requests.__getitem__(key)
            return Workload(arrivals, requests)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return Workload.merge(self, other)

    def __len__(self):
        return len(self.arrivals)

    def __str__(self):
        return (f"Workload(len={len(self)}, "
                f"rate={self.rate:.2f}, "
                f"CV={self.cv:.2f}, "
                f"tstamps={to_str_round(self.arrivals[:20])} ...)")


if __name__ == "__main__":
    w1 = PoissonProcess(10).generate_workload("m", start=0, duration=1000, seed=0)
    w2 = GammaProcess(10, 5).generate_workload("m", start=0, duration=1000, seed=0)

    w3 = w1 + w2
    print(w3)

    ws = w3.split_time_interval(500)
    print(len(ws))
    print(ws[0])
    print(ws[1])
