import math
import os.path
import csv
import pickle
import time

import copy
import warnings
from typing import List, Dict
from collections import OrderedDict

import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, pareto
import numpy as np

from alpa_serve.simulator.workload import Workload, PoissonProcess, GammaProcess, \
    Request, ParetoProcess


DEBUG = False


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
          f"avg: {np.mean(num_function_invocations):.2f}")

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
          f"avg: {np.mean(num_function_invocations):.2f}")

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
    if DEBUG:
        print(f"Trace: {path[:-4]}, stats: #days: 14, #functions: {num_functions}, "
              f"total invocations: {sum(num_function_invocations)}, "
              f"max: {max(num_function_invocations)}, min: {min(num_function_invocations)}, "
              f"avg: {np.mean(num_functions):.2f}")
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
                 arrival_distribution_params=None,
                 rate_scale_factor=1.0,
                 cv_scale_factor=1.0,
                 time_scale_factor=1.0,
                 replication_factor=1):
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

        # scale factors
        self.rate_scale_factor = rate_scale_factor
        self.cv_scale_factor = cv_scale_factor
        self.time_scale_factor = time_scale_factor
        self.replication_factor = replication_factor

        # stats
        if len(self.arrivals) > 1:
            self._rate = len(self.arrivals) / ((self.end_seconds - self.start_seconds) // self.time_scale_factor)
            intervals = self.arrivals[1:] - self.arrivals[:-1]
            self._cv = np.std(intervals)  / (np.mean(intervals) + 1e-5)
        else:
            self._rate = 0
            self._cv = 0

    def to_workload(self, slo: float):
        return Workload(self.arrivals, [Request(self.model, None, slo, i, {})
                                        for i in range(len(self.arrivals))])

    def report_stats(self):
        print(f"Trace for {self.model}, duration: {self.duration}, {self.duration_seconds} (s), #arrivals: {self.arrivals.size}, "
              f"arrival distribution: {self.arrival_distribution}, "
              f"generation interval: {self.interval_seconds}, "
              f"scale factor: ({self.rate_scale_factor}, {self.cv_scale_factor}, {self.time_scale_factor}, {self.replication_factor}). "
              f"overall rate: {self._rate:.2f}, overall cv: {self._cv:.2f}.")


    def visualize(self, n_interval=100):
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
        plt.ylim(0, 200)
        fig = plt.gcf()
        figure_size = (8, 4)
        fig.set_size_inches(figure_size)
        fig_folder = "plots"
        os.makedirs(fig_folder, exist_ok=True)
        fig_name = f"{self.model}-{self.trace_name}-{self.arrival_distribution}-" \
                   f"{self.start_time}-{self.end_time}-{self.interval_seconds}-" \
                   f"({self.rate_scale_factor}, {self.cv_scale_factor}," \
                   f"{self.time_scale_factor}, {self.replication_factor}).png"
        fig.savefig(os.path.join(fig_folder, fig_name), bbox_inches='tight')
        plt.close()

    def rate(self):
        return self._rate

    def cv(self):
        return self._cv

    @property
    def duration_seconds(self):
        duration_seconds = self.end_seconds - self.start_seconds
        return duration_seconds

    @property
    def duration(self):
        duration_mins = self.duration_seconds // 60
        duration_remained_seconds = self.duration_seconds % 60
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


def report_group_stats(replays: List[TraceReplay]):
    n_model = len(replays)
    all_arrivals = np.concatenate([m.arrivals for m in replays])
    arrivals = np.sort(all_arrivals)
    n_arrival = all_arrivals.size
    rate = n_arrival / replays[0].duration_seconds
    intervals = arrivals[1:] - arrivals[:-1]
    cv = np.std(intervals) / (np.mean(intervals) + 1e-5)
    print(
        f"Trace for a group of {n_model} models, duration: {replays[0].duration}, {replays[0].duration_seconds} (s), "
        f"#arrivals: {n_arrival}, arrival distribution: {replays[0].arrival_distribution}, "
        f"generation interval: {replays[0].interval_seconds}, "
        f"Overall cluster rate: {rate:.2f}, cluster cv: {cv:.2f}.")


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
               model_mapping_strategy: str = "stripe",
               start_time: str = "0.0.0",
               end_time: str = "13.23.60",
               arrival_distribution : str = "exponential",
               interval_seconds: int = 600,
               rate_scale_factor: float = 1.0,
               cv_scale_factor: float = 1.0,
               time_scale_factor: float = 1.0,
               replication_factor: int = 1,
               seed: int = 0) -> Dict[str, TraceReplay]:
        """Return a workload that replays a given slice of the trace.

        The method replays the trace by mapping functions in the trace to models provided by
        the input `models`.

        Args:
            models (List[str]): a list of model names.
            model_mapping_strategy (str): `round_robin` or `stripe`.
            start_time (str): in the form of `{day}.{hour}.{minute}`.
            end_time (str): in the form of `{day}.{hour}.{minute}`.
            arrival_distribution (str): `vanilla`, `exponential`, or `gamma`.
            interval_seconds (int): the length of the interval in seconds to estimate a generation process.
            rate_scale_factor (float): scale the estimated rate give this factor.
            cv_scale_factor (float): scale the cv given this factor. Only works when distribution = `gamma`.
            time_scale_factor (float): downscale the time, e.g., when it is 2,
                a 1-hour trace will be used as if it were 30 mins.
            replication_factor (int): simply replicate each arrival given a factor.
            seed (int): random seed for the generation process.

        Returns:
            replays (Dict[str, TraceReplay]): the TraceReplay for each model.
        """
        # Do some checks
        if replication_factor < 1:
            warnings.warn("`replication factor` should not be less than 1. Reset it to 1.")
        if replication_factor > 1:
            if not (self.trace_name == "azure_v2" and arrival_distribution == "vanilla"):
                raise RuntimeError(f"We can only replicate vanilla azure v2 trace, "
                                   f"got: {self.trace_name}, {arrival_distribution}")
        if time_scale_factor != 1.0:
            if self.trace_name != "azure_v2":
                raise RuntimeError("Cannot do time-scaling on azure_v1.")
            if arrival_distribution != "vanilla":
                raise RuntimeError("Can only do time-scaling on vanilla distributions.")
        if arrival_distribution != "gamma" and cv_scale_factor != 1.0:
            raise RuntimeError("No CV for exponential distributions.")
        if time_scale_factor != 1.0 and (rate_scale_factor != 1.0 or cv_scale_factor != 1.0):
            raise RuntimeError("Choose one: scale rate/cv, or scale time.")

        replays = OrderedDict()
        start_d, start_h, start_m = self.timestr_to_dhm(start_time)
        end_d, end_h, end_m = self.timestr_to_dhm(end_time)
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60

        if self.trace_name == "azure_v1":
            # Trace are 1-min histograms
            # 1. Convert function trace to model trace
            model_histogram = OrderedDict()
            function_histogram = self.slice(start_time, end_time)
            # filter out all functions that have zero arrivals:
            functions_to_remove = [f for f in function_histogram if np.sum(function_histogram[f]) == 0]
            for f in functions_to_remove:
                del function_histogram[f]
            # generate function model mapping.
            function_model_mapping = self.map_model(models, function_histogram.keys(), model_mapping_strategy)
            for f, m in function_model_mapping.items():
                if m not in model_histogram:
                    model_histogram[m] = copy.deepcopy(function_histogram[f])
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
                                                                    arrival_distribution,
                                                                    rate_scale_factor,
                                                                    cv_scale_factor)
        elif self.trace_name == "azure_v2":
            # Trace are exact arrivals
            # 1. Convert function trace to model trace
            model_arrivals = OrderedDict()
            assert self.function_arrivals is not None
            function_arrivals = self.slice(start_time, end_time)
            functions_to_remove = [f for f in function_arrivals if function_arrivals[f].size == 0]
            for f in functions_to_remove:
                del function_arrivals[f]
            # generate function model mapping.
            function_model_mapping = self.map_model(models, function_arrivals.keys(), model_mapping_strategy)
            for f, m in function_model_mapping.items():
                if m not in model_arrivals:
                    model_arrivals[m] = function_arrivals[f]
                else:
                    model_arrivals[m] = np.concatenate((model_arrivals[m], function_arrivals[f]))
            for m in model_arrivals:
                model_arrivals[m] = np.sort(model_arrivals[m])

            if arrival_distribution == "vanilla":
                if replication_factor > 1:
                    for m in model_arrivals:
                        model_arrivals[m] = np.repeat(model_arrivals[m], replication_factor)
                for m in model_arrivals:
                    model_arrivals[m] = (model_arrivals[m] - start_timestamp_seconds) / time_scale_factor + start_timestamp_seconds
                    replays[m] = TraceReplay(m,
                                        model_arrivals[m],
                                        self.trace_name,
                                        start_time,
                                        end_time,
                                        end_timestamp_seconds - start_timestamp_seconds,
                                        arrival_distribution,
                                        rate_scale_factor=rate_scale_factor,
                                        cv_scale_factor=cv_scale_factor,
                                        time_scale_factor=time_scale_factor,
                                        replication_factor=replication_factor)
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
            distributions = self.estimate_parameters_with_arrivals(arrival_dataset,
                                                                   arrival_distribution,
                                                                   interval_seconds,
                                                                   rate_scale_factor,
                                                                   cv_scale_factor)
        else:
            raise NotImplementedError("Other trace ")

        # Sample from the distributions and generate the arrivals
        for m in distributions:
            arrivals = []
            arrival_distribution_params = []
            for i, distribution in enumerate(distributions[m]):
                if distribution is None:
                    arrival_distribution_params.append(None)
                    continue
                start = i * interval_seconds + start_timestamp_seconds
                arrivals.extend(distribution.generate_arrivals(start, interval_seconds, seed))
                # if DEBUG:
                #     arrivals.extend(distribution.generate_arrivals(0, 1.0e9, seed))
                #     self.visualize_inter_arrival(np.array(arrivals), "test")
                arrival_distribution_params.append(distribution.params())
                seed += 1
            replays[m] = TraceReplay(m,
                                     np.array(arrivals),
                                     self.trace_name,
                                     start_time,
                                     end_time,
                                     interval_seconds,
                                     arrival_distribution,
                                     arrival_distribution_params=arrival_distribution_params,
                                     rate_scale_factor=rate_scale_factor,
                                     cv_scale_factor=cv_scale_factor,
                                     time_scale_factor=time_scale_factor)

        return replays

        # sort models
        # keys = list(replays.keys())
        # num_models = len(models)
        # indices = list(range(num_models))
        # indices.sort(key=lambda i: -len(replays[keys[i]].arrivals))

        # new_replay = OrderedDict()
        # for i in range(num_models):
        #     new_replay[models[i]] = replays[keys[indices[i]]]
        #     new_replay[models[i]].model = models[i]

        # return new_replay

    def replay_vanilla(self,
                       models: List[str],
                       model_mapping_strategy: str ="stripe",
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

    def map_model(self, models, function_names, strategy="stripe"):
        mapping = OrderedDict()
        n_model = len(models)
        n_function = len(function_names)
        assert n_function >= n_model, f"#function {n_function} < #models {n_model}"
        if strategy not in ["round_robin", "stripe"]:
            raise NotImplementedError(f"Unimplemented strategy: {strategy}")
        for i, f in enumerate(function_names):
            if strategy == "round_robin":
                mapping[f] = models[n_model * i // n_function]
            else:
                mapping[f] = models[i % n_model]
        return mapping

    def estimate_parameters_with_histogram(self,
                                           dataset,
                                           interval_seconds,
                                           arrival_distribution="exponential",
                                           rate_scale_factor=1.0,
                                           cv_scale_factor=1.0):
        if arrival_distribution not in ["exponential", "gamma"]:
            raise NotImplementedError(f"We can only use histogram data for exponential or gamma distribution, "
                                      f"got {arrival_distribution}")
        distributions = OrderedDict()
        for model, histogram in dataset.items():
            distributions[model] = []
            for h in histogram:
                if h == 0:
                    distributions[model].append(None)
                else:
                    arrival_rate = h / interval_seconds
                    arrival_rate = arrival_rate * rate_scale_factor
                    if arrival_distribution == "exponential":
                        distributions[model].append(PoissonProcess(arrival_rate))
                    else:
                        distributions[model].append(GammaProcess(arrival_rate, cv_scale_factor))
        return distributions

    def estimate_parameters_with_arrivals(self,
                                          dataset,
                                          arrival_distribution="exponential",
                                          interval_seconds=600,
                                          rate_scale_factor=1.0,
                                          cv_scale_factor=1.0):
        if arrival_distribution not in ["exponential", "gamma", "pareto"]:
            raise NotImplementedError(f"Only support exponential | gamma | pareto, "
                                      f" got {arrival_distribution}")
        distributions = OrderedDict()
        model_index = 0
        for model, arrivals in dataset.items():
            distributions[model] = []
            for i, arrival in enumerate(arrivals):
                empirical_arrival_rate = arrival.size / interval_seconds
                inter_arrival = np.diff(arrival) + 1e-6
                if inter_arrival.size == 0 or (inter_arrival.size == 1 and arrival_distribution == "gamma"):
                    distributions[model].append(None)
                else:
                    if DEBUG:
                        self.visualize_inter_arrival(inter_arrival, f"{model}-{i}", n_interval=2000)
                    if arrival_distribution == "exponential":
                        arrival_rate = self.estimate_exponential(inter_arrival)
                        if np.isnan(arrival_rate):
                            distributions[model].append(None)
                            continue
                        if arrival_rate > 5 * empirical_arrival_rate:
                            if DEBUG:
                                warnings.warn(f"Estimation for model {model_index} is highly biased. "
                                              f"Hard reset to empirical rate: {empirical_arrival_rate}.")
                            arrival_rate = empirical_arrival_rate
                        arrival_rate *= rate_scale_factor
                        distributions[model].append(PoissonProcess(arrival_rate))
                    elif arrival_distribution == "gamma":
                        try:
                            arrival_rate, cv = self.estimate_gamma(inter_arrival)
                            if np.isnan(arrival_rate) or np.isnan(cv):
                                distributions[model].append(None)
                                continue
                            if arrival_rate > 5 * empirical_arrival_rate:
                                if DEBUG:
                                    warnings.warn(f"Estimation for model {model_index} is highly biased. "
                                                  f"Hard reset to empirical rate: {empirical_arrival_rate}.")
                                arrival_rate = empirical_arrival_rate
                            # scale them
                            arrival_rate *= rate_scale_factor
                            cv *= cv_scale_factor
                            distributions[model].append(GammaProcess(arrival_rate, cv))
                        except ValueError as ve:
                            warnings.warn("Failed to fit a gamma distribution.")
                            distributions[model].append(None)
                    elif arrival_distribution == "pareto":
                        inter_arrival += 1.0
                        shape, scale, loc = self.estimate_pareto(inter_arrival)
                        if np.isnan(shape) or np.isnan(scale) or np.isnan(loc):
                            continue
                        distributions[model].append(ParetoProcess(shape, scale, loc))
                    else:
                        raise RuntimeError(f"Unrecognized distribution: {arrival_distribution}")
            model_index += 1
        return distributions

    @staticmethod
    def visualize_inter_arrival(inter_arrival, name, n_interval=300):
        count, bins, _ = plt.hist(inter_arrival, bins=np.linspace(0, 300, n_interval))
        plt.show()
        plt.ylabel("#reqs")
        plt.xlabel("#seconds")
        fig = plt.gcf()
        figure_size = (8, 4)
        fig.set_size_inches(figure_size)
        fig.savefig(f"plots/{name}.png", bbox_inches='tight')
        plt.close()

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
    def estimate_pareto(inter_arrivals):
        shape, loc, scale = pareto.fit(inter_arrivals, floc=0.0, fscale=1.0)
        return shape, scale, loc

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
              f"avg: {np.mean(n_invocations):.2f}")

    def bic(self):
        pass

    def aic(self):
        pass
