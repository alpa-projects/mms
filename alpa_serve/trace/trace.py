import math
import os.path
import csv
import pickle
import time
import warnings
from typing import List
from collections import OrderedDict

import matplotlib.pyplot as plt
from scipy.stats import expon, gamma
import numpy as np

from alpa_serve.simulator.workload import Workload, PoissonProcess, GammaProcess, Request


def preprocess_azure_v1_trace(trace_dir, n_day=14):
    if not os.path.exists(trace_dir):
        raise RuntimeError(f"{trace_dir}")
    tracelines = OrderedDict()
    print(f"Reading azure v1 trace in 14 days; it might take a while...")
    tic = time.time()
    for i in range(1, n_day + 1):
        day_str = str(i) if i >= 10 else "0" + str(i)
        filename = os.path.join(trace_dir, f"invocations_per_function_md.anon.d{day_str}.csv")
        print(f"Read file: {filename}")
        with open(filename, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                function_name = row["HashFunction"]
                histogram_1min = np.array([int(row[str(j)]) for j in range(1, 1441)], dtype=np.int32)
                if i == 1:
                    assert function_name not in tracelines
                    tracelines[function_name] = histogram_1min
                else:
                    expected_size = 1440 * (i - 1)
                    if function_name in tracelines:
                        cur_size = tracelines[function_name].size
                        if cur_size != expected_size:
                            diff = expected_size - cur_size
                            assert diff % 1440 == 0
                            tracelines[function_name] = np.concatenate((tracelines[function_name],
                                                                       np.zeros((diff,), dtype=np.int32),
                                                                       histogram_1min))
                        else:
                            tracelines[function_name] = np.concatenate((tracelines[function_name],
                                                                       histogram_1min))
                    else:
                        tracelines[function_name] = np.concatenate((np.zeros((expected_size, ), dtype=np.int32),
                                                                   histogram_1min))
    for function_name, histogram_1min in tracelines.items():
        if histogram_1min.size != n_day * 1440:
            diff = n_day * 1440 - histogram_1min.size
            assert diff % 1440 == 0
            tracelines[function_name] = np.concatenate((tracelines[function_name], np.zeros((diff,), dtype=np.int32)))
    print(f"Reading takes: {time.time() - tic}s.")

    # report the stats.
    num_function_invocations = []
    for function_name, histogram_1min in tracelines.items():
        assert (histogram_1min.size == 1440 * n_day), f"length: {histogram_1min.size}"
        num_function_invocations.append(np.sum(histogram_1min))
    num_functions = len(tracelines.keys())
    print(f"Azure trace v1, stats: #days: {n_day}, #functions: {num_functions}, "
          f"total invocations: {sum(num_function_invocations)}, "
          f"max: {max(num_function_invocations)}, min: {min(num_function_invocations)}, "
          f"avg: {sum(num_function_invocations) / num_functions}")

    # pickle it to disk
    save_path = os.path.join(trace_dir, "azure_v1.pkl")
    with open(save_path, "wb") as handle:
        pickle.dump(tracelines, handle)
    print(f"Dump the data into {save_path}, file size: {os.path.getsize(save_path) // 1e6} MB.")


def preprocess_azure_v2_trace(trace_dir):
    """Load and process azure v2 trace."""
    if not os.path.exists(trace_dir):
        raise RuntimeError(f"{trace_dir}")
    filename = os.path.join(trace_dir, "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
    tracelines = OrderedDict()
    print(f"Reading azure v2 trace in 14 days...")
    tic = time.time()
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            function_name = row["func"]
            end_time = float(row["end_timestamp"])
            duration = float(row["duration"])
            if function_name not in tracelines:
                tracelines[function_name] = [end_time - duration]
            else:
                tracelines[function_name].append(end_time -duration)

    for function_name, trace in tracelines.items():
        tracelines[function_name] = np.sort(np.array(tracelines[function_name]))
    print(f"Reading takes: {time.time() - tic}s.")
    # Do some check and report stats:
    num_functions = len(tracelines.keys())
    num_function_invocations = []
    for function_name, trace in tracelines.items():
        num_function_invocations.append(len(trace))
    print(f"Azure trace v2, stats: #days: 14, #functions: {num_functions}, "
          f"total invocations: {sum(num_function_invocations)}, "
          f"max: {max(num_function_invocations)}, min: {min(num_function_invocations)}, "
          f"avg: {sum(num_function_invocations) / num_functions}")

    # pickle it to disk
    save_path = os.path.join(trace_dir, "azure_v2.pkl")
    with open(save_path, "wb") as handle:
        pickle.dump(tracelines, handle)
    print(f"Dump the data into {save_path}, file size: {os.path.getsize(save_path) // 1e6} MB.")


def load_trace(path: str) -> OrderedDict:
    assert path.endswith(".pkl")
    tic = time.time()
    with open(path, "rb") as handle:
        tracelines = pickle.load(handle)
    print(f"Reading takes: {time.time() - tic}s.")

    # Do some check and report stats:
    num_functions = len(tracelines.keys())
    num_function_invocations = []
    for function_name, trace in tracelines.items():
        if trace.dtype == np.int32:
            num_function_invocations.append(np.sum(trace))
        else:
            num_function_invocations.append(trace.size)

    print(f"Trace: {path[:-4]}, stats: #days: 14, #functions: {num_functions}, "
          f"total invocations: {sum(num_function_invocations)}, "
          f"max: {max(num_function_invocations)}, min: {min(num_function_invocations)}, "
          f"avg: {sum(num_function_invocations) / num_functions}")
    return tracelines


class TraceReplay:
    def __init__(self,
                 model,
                 arrivals,
                 trace_name,
                 start_time,
                 end_time,
                 interval_seconds,
                 arrival_distribution,
                 arrival_distribution_params=None):
        """A TraceReplay specifies the traffic arrival pattern of a model."""
        self.model = model
        self.arrivals = arrivals

        # other generation-time information
        self.trace_name = trace_name
        self.start_time = start_time
        self.end_time = end_time
        self.arrival_distribution = arrival_distribution
        self.interval_seconds = interval_seconds
        self.arrival_distribution_params = arrival_distribution_params

    def to_workload(self, slo: float):
        return Workload(self.arrivals.tolist(), [Request(self.model, None, slo, i, {})
                                        for i in range(len(self.arrivals))])

    def report_stats(self):
        if self.arrival_distribution_params is not None:
            rates = []
            cvs = []
            for param in self.arrival_distribution_params:
                if param is not None:
                    rate, cv = param
                    rates.append(rate)
                    cvs.append(cv)
            print(f"Trace for model: {self.model}, duration: {self.duration}, "
                  f"duration (seconds): {self.duration_seconds}, "
                  f"arrival distribution: {self.arrival_distribution}, "
                  f"generation interval: {self.interval_seconds}, "
                  f"generation rates: min rate {min(rates):.2f}, mean rate {sum(rates) / len(rates):.2f}, max rate: {max(rates):.2f}, "
                  f"generate cvs: min cv {min(cvs):.2f}, mean cv {sum(cvs) / len(cvs):.2f}, max cv: {max(cvs):.2f}, "
                  f"#arrivals: {self.arrivals.size}")
        else:
            print(f"Trace for model: {self.model}, duration: {self.duration}, "
                  f"duration (seconds): {self.duration_seconds}, "
                  f"arrival distribution: {self.arrival_distribution}, "
                  f"generation interval: {self.interval_seconds}, "
                  f"n arrivals: {self.arrivals.size}")

    def visualize(self, n_interval=100):
        if np.argwhere(np.isnan(self.arrivals)).size > 0:
            print(self.arrivals)
        assert np.all(self.arrivals > self.start_seconds), \
            f"arrivals: {np.argwhere(np.isnan(self.arrivals))}, " \
            f"start_seconds: {self.start_seconds}"
        plt.figure()
        plt.hist(self.arrivals, bins=np.linspace(self.start_seconds, self.end_seconds,
                                                 n_interval),
                 alpha=0.8, label=self.model)
        plt.title("Sample requests histogram (bin size = 0.1s)")
        plt.ylabel("#requests")
        plt.xlabel("time (s)")
        plt.legend()
        plt.ylim(0, 1000)
        fig = plt.gcf()
        figure_size = (8, 4)
        fig.set_size_inches(figure_size)
        fig_folder = "plots"
        os.makedirs(fig_folder, exist_ok=True)
        fig_name = f"{self.model}-{self.trace_name}-{self.arrival_distribution}-" \
                   f"{self.start_time}-{self.end_time}-{self.interval_seconds}.png"
        fig.savefig(os.path.join(fig_folder, fig_name), bbox_inches='tight')
        plt.close()

    @property
    def n_arrivals(self):
        return self.arrivals.size

    @property
    def duration_seconds(self):
        duration_seconds = self.end_seconds - self.start_seconds
        return duration_seconds

    @property
    def duration(self):
        duration_mins = self.duration_seconds // 60
        duration_remained_seconds = duration_mins % 60
        duration_hours = duration_mins // 60
        duration_remained_mins = duration_mins % 60
        duration_day = duration_hours // 24
        duration_remained_hours = duration_hours % 24
        return duration_day, duration_remained_hours, duration_remained_mins, duration_remained_seconds

    @property
    def start_seconds(self):
        start_d, start_h, start_m = Trace.timestr_to_dhm(self.start_time)
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        return start_timestamp_seconds

    @property
    def end_seconds(self):
        end_d, end_h, end_m = Trace.timestr_to_dhm(self.end_time)
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        return end_timestamp_seconds


class Trace:
    def __init__(self, trace_name, trace_dir):
        self.trace_name: str = trace_name
        self.trace_dir: str = trace_dir
        self.have_timestamp: bool = False
        self.function_arrivals = None
        self.function_histogram = None
        self.n_day = 14

        if trace_name == "azure_v1":
            self.function_histogram = load_trace(trace_dir)
        elif trace_name == "azure_v2":
            self.function_arrivals = load_trace(trace_dir)
        elif trace_name == "alibaba":
            raise NotImplementedError(f"To be implemented for {trace_name}")
        else:
            raise RuntimeError("Choose trace from `azure_v1 | azure_v2 | alibaba`")

    @staticmethod
    def timestr_to_dhm(time_str):
        dhm = time_str.split(sep=".")
        if len(dhm) != 3:
            raise RuntimeError("Wrong format for `start_time`.")
        day = int(dhm[0])
        hour = int(dhm[1])
        min = int(dhm[2])
        return day, hour, min

    def slice(self, start_time: str = "0.0.0", end_time: str = "13.23.60"):
        """Slice the trace given start time string and end_time string."""
        start_d, start_h, start_m = self.timestr_to_dhm(start_time)
        end_d, end_h, end_m = self.timestr_to_dhm(end_time)
        if start_d >= self.n_day or end_d >= self.n_day or start_d > end_d:
            raise RuntimeError("start day or end day must be within the trace range.")
        if start_h >= 24 or end_h >= 24:
            raise RuntimeError("start hour or end hour must be < 24.")
        if start_m > 60 or end_m > 60:
            raise RuntimeError("start min or end minute must be <= 60.")
        if self.trace_name == "azure_v1":
            ret =  self.slice_histogram(start_d, start_h, start_m, end_d, end_h, end_m)
        elif self.trace_name == "azure_v2":
            ret =  self.slice_arrival(start_d, start_h, start_m, end_d, end_h, end_m)
        else:
            raise NotImplementedError()
        self.report_stats(ret)
        return ret

    def slice_histogram(self, start_d, start_h, start_m, end_d, end_h, end_m):
        """Slice the histogram."""
        assert self.function_histogram is not None
        start_slot = start_d * 24 * 60 + start_h * 60 + start_m
        end_slot = end_d * 24 * 60 + end_h * 60 + end_m
        sliced_histogram = OrderedDict()
        for function_name, histogram in self.function_histogram.items():
            sliced_histogram[function_name] = histogram[start_slot:end_slot]
        return sliced_histogram

    def slice_arrival(self, start_d, start_h, start_m, end_d, end_h, end_m):
        assert self.function_arrivals is not None
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        sliced_arrival = OrderedDict()
        for function_name, trace in self.function_arrivals.items():
            tmp = trace[trace >= start_timestamp_seconds]
            tmp = tmp[tmp < end_timestamp_seconds]
            sliced_arrival[function_name] = tmp
        return sliced_arrival

    def replay(self,
               models: List[str],
               model_mapping_strategy = "round_robin",
               start_time: str = "0.0.0",
               end_time: str = "13.23.60",
               arrival_distribution="exponential",
               interval_seconds: int = 60):
        """Return a workload that replays a given slice of the trace."""
        replays = OrderedDict()
        # Step 1: generate the model-function mapping
        function_model_mapping = self.map_model(models, self.function_names, model_mapping_strategy)
        start_d, start_h, start_m = self.timestr_to_dhm(start_time)
        end_d, end_h, end_m = self.timestr_to_dhm(end_time)
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60

        if self.trace_name == "azure_v1":
            # Trace are 1-min histograms
            # 1. Convert function trace to model trace
            model_histogram = OrderedDict()
            function_histogram = self.slice(start_time, end_time)
            for f, m in function_model_mapping.items():
                if m not in model_histogram:
                    model_histogram[m] = function_histogram[f]
                else:
                    model_histogram[m] += function_histogram[f]

            # 2. re-histogram based on `interval_seconds`
            histogram_dataset = OrderedDict()
            assert interval_seconds % 60 == 0, "Please set `interval_seconds` as a multiple of 60"
            n_min_per_interval = interval_seconds // 60
            for model, histogram in model_histogram.items():
                n_total_min = histogram.size
                n_interval = (n_total_min + n_min_per_interval - 1) // n_min_per_interval
                accumulated = np.zeros((n_interval,), dtype=np.int32)
                for i in range(accumulated.size):
                    start = i * n_min_per_interval
                    end = (i + 1) * n_min_per_interval if (i + 1) * n_min_per_interval <= n_total_min else n_total_min
                    accumulated[i] = np.sum(histogram[start:end])
                histogram_dataset[model] = accumulated

            # Estimate distribution parameters with histogram dataset
            distributions = self.estimate_parameters_with_histogram(histogram_dataset,
                                                                    interval_seconds,
                                                                    arrival_distribution)
        elif self.trace_name == "azure_v2":
            # Trace are exact arrivals
            # 1. Convert function trace to model trace
            model_arrivals = OrderedDict()
            assert self.function_arrivals is not None
            function_arrivals = self.slice(start_time, end_time)
            for f, m in function_model_mapping.items():
                if m not in model_arrivals:
                    model_arrivals[m] = function_arrivals[f]
                else:
                    model_arrivals[m] = np.concatenate((model_arrivals[m], function_arrivals[f]))
            for m in model_arrivals:
                model_arrivals[m] = np.sort(model_arrivals[m])


            if arrival_distribution == "vanilla":
                for m in model_arrivals:
                    # TODO: Change this to be workload instead TraceReplay?
                    replays[m] = (
                        TraceReplay(m, model_arrivals[m], self.trace_name, start_time, end_time,
                                    end_timestamp_seconds - start_timestamp_seconds, arrival_distribution, None))
                return replays

            # 2. bucketing arrivals based on `interval_seconds` and start/end time.
            arrival_dataset = OrderedDict()
            intervals = np.arange(start_timestamp_seconds, end_timestamp_seconds, interval_seconds)
            if intervals[-1] != end_timestamp_seconds:
                intervals = np.append(intervals, end_timestamp_seconds)
            for m in model_arrivals:
                arrivals = model_arrivals[m]
                interval_dataset = []
                for i in range(intervals.size - 1):
                    tmp = arrivals[arrivals >= intervals[i]]
                    tmp = tmp[tmp < intervals[i+1]]
                    interval_dataset.append(tmp)
                arrival_dataset[m] = interval_dataset

            # 3. estimate distribution parameters based on arrivals
            distributions = self.estimate_parameters_with_arrivals(arrival_dataset, arrival_distribution)
        else:
            raise NotImplementedError("Other trace ")

        # Sample from the distributions and generate the arrivals
        for m in distributions:
            arrivals = []
            arrival_distribution_params = []
            for seed, distribution in enumerate(distributions[m]):
                if distribution is None:
                    arrival_distribution_params.append(None)
                    continue
                start = seed * interval_seconds + start_timestamp_seconds
                generated = distribution.generate_arrivals(start, interval_seconds, seed)
                arrivals.extend(distribution.generate_arrivals(start, interval_seconds, seed))
                arrival_distribution_params.append(distribution.params())
            replays[m] = TraceReplay(m, np.array(arrivals), self.trace_name, start_time, end_time,
                                     interval_seconds, arrival_distribution, arrival_distribution_params)
        return replays

    def replay_vanilla(self,
                       models: List[str],
                       model_mapping_strategy: str ="round_robin",
                       start_time: str = "0.0.0",
                       end_time: str = "13.23.60"):
        """Return exactly the same trace; only works for azure_v2."""
        if self.trace_name == "azure_v1":
            raise RuntimeError(f"Cannot replay {self.trace_name} trace in vanilla.")
        return self.replay(models,
                           model_mapping_strategy,
                           start_time=start_time,
                           end_time=end_time,
                           arrival_distribution="vanilla")

    def map_model(self, models, function_names, strategy="round_robin"):
        mapping = OrderedDict()
        n_model = len(models)
        n_function = len(function_names)
        assert n_function >= n_model
        if strategy not in ["round_robin", "stripe"]:
            raise NotImplementedError(f"Unimplemented strategy: {strategy}")
        for i, f in enumerate(function_names):
            if strategy == "round_robin":
                mapping[f] = models[n_model * i // n_function]
            else:
                mapping[f] = models[i % n_model]
        return mapping

    def estimate_parameters_with_histogram(self, dataset, interval_seconds, arrival_distribution="exponential"):
        if arrival_distribution not in ["exponential"]:
            raise NotImplementedError(f"We can only use histogram data for exponential distribution,"
                                      f" got {arrival_distribution}")
        distributions = OrderedDict()
        for model, histogram in dataset.items():
            distributions[model] = []
            for h in histogram:
                if h == 0:
                    distributions[model].append(PoissonProcess(arrival_rate))
                else:
                    arrival_rate = h / interval_seconds
                    distributions[model].append(PoissonProcess(arrival_rate))
        return distributions

    def estimate_parameters_with_arrivals(self, dataset, arrival_distribution="exponential"):
        if arrival_distribution not in ["exponential", "gamma"]:
            raise NotImplementedError(f"Only support exponential | gamma, "
                                      f" got {arrival_distribution}")
        distributions = OrderedDict()
        for model, arrivals in dataset.items():
            distributions[model] = []
            for arrival in arrivals:
                inter_arrival = np.diff(arrival) + 1e-6
                if inter_arrival.size == 0 or (inter_arrival.size == 1 and arrival_distribution == "gamma"):
                    distributions[model].append(None)
                else:
                    if arrival_distribution == "exponential":
                        arrival_rate = self.estimate_exponential(inter_arrival)
                        distributions[model].append(PoissonProcess(arrival_rate))
                    else:
                        try:
                            arrival_rate, cv = self.estimate_gamma(inter_arrival)
                            distributions[model].append(GammaProcess(arrival_rate, cv))
                        except ValueError as ve:
                            warnings.warn("Failed to fit a gamma distribution.")
                            distributions[model].append(None)
        return distributions

    @staticmethod
    def estimate_exponential(inter_arrivals):
        """Take inter-arrivals and return the rate parameters."""
        _, scale = expon.fit(inter_arrivals, floc=0)
        return 1.0 / scale

    @staticmethod
    def estimate_gamma(inter_arrivals):
        shape, _, scale = gamma.fit(inter_arrivals, floc=0)
        cv = math.sqrt(1.0 / shape)
        arrival_rate = 1.0 / (shape * scale)
        return arrival_rate, cv

    @staticmethod
    def estimate_mmpp(self, inter_arrivals):
        pass

    @property
    def function_names(self):
        if self.trace_name == "azure_v1":
            return list(self.function_histogram.keys())
        elif self.trace_name == "azure_v2":
            return list(self.function_arrivals.keys())
        else:
            raise NotImplementedError()

    @staticmethod
    def report_stats(trace: OrderedDict):
        n_invocations = []
        n_function = len(trace.keys())
        for function_name, arrival_or_histogram in trace.items():
            if arrival_or_histogram.dtype == np.int32:
                n_invocations.append(np.sum(arrival_or_histogram))
            else:
                n_invocations.append(arrival_or_histogram.size)
        print(f"Sliced trace stats: #functions: {n_function}, "
          f"total invocations: {sum(n_invocations)}, "
          f"max: {max(n_invocations)}, min: {min(n_invocations)}, "
          f"avg: {sum(n_invocations) / n_function}")
