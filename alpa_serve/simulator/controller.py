"""
The serving controller.

This file simulates `alpa_serve/controller.py`.
"""
import asyncio
import math
from collections import defaultdict
from functools import partial
from typing import Callable, List, Dict, Optional, Tuple

import numpy as np

from alpa_serve.controller import CreateInfo, ModelInfo, GroupInfo, build_logger
from alpa_serve.profiling import ProfilingResult
from alpa_serve.simulator.cluster import VirtualMesh
from alpa_serve.simulator.event_loop import (timed_coroutine, clock,
    main_loop, sleep, run_event_loop)
from alpa_serve.simulator.util import install_remote_methods, async_to_sync
from alpa_serve.simulator.workload import Workload, StatsResult, PerDeviceStatsResult
from alpa_serve.util import ServingCase
from alpa.util import to_str_round


class GroupManager:
    """
    Simulates alpa_serve/controller.py::GroupManager

    This class copies most of the code from the real class.
    """

    def __init__(self, virtual_mesh_shape):
        self.virtual_mesh = VirtualMesh(virtual_mesh_shape)

        # Dict[str -> object]
        self.replicas = {}

        # Dict[model_name -> Dict[batch_size -> List[stage_latency]]]
        self.latency_dict = defaultdict(dict)

        self.stage_clock = [0] * np.prod(virtual_mesh_shape)

        self.logger = build_logger("group_manager")

        # Constants
        self.fixed_overhead = 0.004

        self.alpa_overhead = partial(np.random.normal, loc=0.004, scale=0.001)

        # Simulator specific code
        install_remote_methods(self)

    def create_replica(self, name: str, create_info: CreateInfo):
        assert name not in self.replicas

        model_def, args, kwargs = (create_info.model_def, create_info.init_args,
                                   create_info.init_kwargs)
        args = args or []
        kwargs = kwargs or {}
        kwargs["virtual_mesh"] = self.virtual_mesh
        self.replicas[name] = model_def(*args, **kwargs)

        if hasattr(self.replicas[name], "get_latency_dict"):
            self.latency_dict[name] = self.replicas[name].get_latency_dict()
        else:
            self.latency_dict[name] = defaultdict(lambda: [0])

    @timed_coroutine
    async def handle_request(self, name: str, request):
        request.time_stamp["b"] = clock()

        if request.slo is not None:
            # SLO awareness
            stage_latency = self.latency_dict[name][1]

            # Simulate clock
            req_stage_clock = []
            t = clock()
            for i in range(len(stage_latency)):
                t = max(self.stage_clock[i], t) + stage_latency[i]
                req_stage_clock.append(t)
            ret_time = req_stage_clock[-1]

            # Drop this request if it will exceed deadline
            if ret_time + self.fixed_overhead > request.submit_time + request.slo:
                return None

            # Accept this request
            for i in range(len(stage_latency)):
                self.stage_clock[i] = req_stage_clock[i]

        ret = await self.replicas[name].handle_request(request, delay=self.alpa_overhead())
        return ret


class Controller:
    """
    Simulates alpa_serve/controller.py::Controller

    This class copies most of the code from the real class.
    """

    def __init__(self):
        # Controller metadata
        self.manager_lock = defaultdict(asyncio.Lock)

        # Dict[str -> ModelInfo]
        self.model_info = {}
        # Dict[int -> GroupInfo]
        self.group_info = {}

        self.logger = build_logger("controller")

        # Simulator specific code
        np.random.seed(1)
        self.dispatch_overhead = partial(np.random.normal, loc=0.002, scale=0.0015)

        install_remote_methods(self)

        group_manager_init = partial(lambda: None)
        group_manager_init.remote = partial(GroupManager)
        self.group_manager_class = partial(lambda: None)
        self.group_manager_class.options = lambda *args, **kwargs: group_manager_init

    def create_mesh_group_manager(
            self,
            group_id: int,
            virtual_mesh_shape: Optional[Tuple[int]] = None,
            num_gpus: int = 0):
        assert group_id not in self.group_info, (
            f"Mesh group {group_id} is already launched")
        self.logger.debug(f"Create mesh group manager {group_id} with "
                         f"shape={virtual_mesh_shape}")
        manager = (self.group_manager_class.options(
            name=f"mesh_group_manager_{group_id}",
            num_gpus=num_gpus).remote(virtual_mesh_shape))
        self.group_info[group_id] = GroupInfo(
            manager=manager, queue_size=0, num_total_requests=0)

    def register_model(self,
                       name: str,
                       model_def: Callable,
                       init_args: Optional[List] = None,
                       init_kwargs: Optional[Dict] = None,
                       override: bool = False):
        if name in self.model_info:
            if override:
                for group_id in self.model_info[name].group_ids:
                    self.group_info[group_id].manager.delete_replica.remote(name)
            else:
                raise ValueError(f"Model {name} is already registered")

        self.model_info[name] = ModelInfo(
            CreateInfo(model_def, init_args, init_kwargs), [])

    def create_replica(self,
                       name: str,
                       group_id: int,
                       append_init_args: Optional[List] = None,
                       append_init_kwargs: Optional[Dict] = None):
        assert group_id in self.group_info, (
            f"Group {group_id} does not exist")
        model_info = self.model_info[name]
        manager = self.group_info[group_id].manager
        assert group_id not in model_info.group_ids, (
            f"Model {name} is already created on group {group_id}")
        create_info = model_info.create_info.append_init_args(
            append_init_args, append_init_kwargs)

        self.logger.info(f"Create replica of {name} on group {group_id}")
        model_info.group_ids.append(group_id)
        manager.create_replica.remote(name, create_info)

    def select_group_id(self, group_ids):
        min_id = -1
        min_size = math.inf
        for group_id in group_ids:
            if self.group_info[group_id].queue_size < min_size:
                min_size = self.group_info[group_id].queue_size
                min_id = group_id
        assert min_id != -1
        return min_id

    @timed_coroutine
    async def handle_request(self, request):
        request.time_stamp["a"] = clock()
        name = request.model_name

        assert name in self.model_info, (
            f"Model '{name}' is not registered.")
        model_info = self.model_info[name]

        if not model_info.group_ids:
            return None

        # Dispatch
        group_id = self.select_group_id(model_info.group_ids)
        manager = self.group_info[group_id].manager

        self.group_info[group_id].queue_size += 1
        response = await manager.handle_request.remote(name, request,
            delay=abs(self.dispatch_overhead()))
        self.group_info[group_id].queue_size -= 1
        self.group_info[group_id].num_total_requests += 1

        return response

    def sync(self):
        pass


class Client:
    def __init__(self, controller, debug=False):
        self.controller = controller
        self.debug = debug

        self.res_dict = dict()
        self.http_overhead = partial(np.random.normal, loc=0.0023, scale=0.0005)

    @timed_coroutine
    async def submit_one(self, request, idx, start, finish, good):
        start[idx] = clock()
        request.submit_time = start[idx]
        res = await self.controller.handle_request(request, delay=self.http_overhead())
        finish[idx] = clock()
        e2e_latency = finish[idx] - start[idx]
        good[idx] = e2e_latency <= request.slo and res is not None

        if self.debug:
            tstamps = to_str_round({x: (y - request.submit_time) * 1e3 for x, y in request.time_stamp.items()}, 2)
            print(f"idx: {idx} ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms", flush=True)

    def submit_workload(self, workload: Workload):
        start, finish, good = (np.zeros(len(workload)),
            np.zeros(len(workload)), np.zeros((len(workload),), dtype=np.bool))
        self.res_dict[workload] = (start, finish, good)

        for i in range(len(workload)):
            self.submit_one(workload.requests[i], i, start, finish, good,
                            tstamp=workload.arrivals[i])

    def wait_all(self):
        return main_loop()

    def compute_stats(self, workload: Workload, warmup: float):
        start, finish, good = self.res_dict[workload]
        return workload.compute_stats(start, finish, good, warmup)


async def run_workload(client, workload, warmup):
    client.submit_workload(workload)

    await client.wait_all()

    return client.compute_stats(workload, warmup=warmup)


def simulate_one_case(case: ServingCase, warmup=10, debug=False):
    """Simulate a serving case."""
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = Controller()
    register_models(controller)
    placement = place_models(controller)

    # Launch the client
    client = Client(controller, debug=debug)
    workload = generate_workload()

    # Run workloads
    stats = run_event_loop(run_workload(client, workload, warmup))
    stats.per_device_stats = tuple(
        PerDeviceStatsResult(x.num_total_requests)
        for x in controller.group_info.values()
    )
    return stats, placement


class DummyController:
    """A dummy controller used for approximation with queueing theory."""

    def __init__(self):
        self.name2profiling = {}

        install_remote_methods(self)

    def register_model(self, name: str, model_def: Callable):
        assert isinstance(model_def, partial)

        for a in model_def.args:
            if isinstance(a, ProfilingResult):
                self.name2profiling[name] = a
                break

    def create_mesh_group_manager(self, *args, **kwargs):
        pass

    def create_replica(self, *args, **kwargs):
        pass

    def sync(self, *args, **kwargs):
        pass


def kingman_formula(arrival_rate, arrival_CV, service_rate):
    p = arrival_rate / service_rate
    if p > 1:
        return 100
    assert 0 <= p <= 1
    return p / (1 - p) * (arrival_CV ** 2) / 2 * (1 / service_rate)


def approximate_one_case(case: ServingCase,
                         seed: int = 0,
                         warmup: int = 10,
                         debug: bool = False):
    """Use kingman's formula to approximate a case."""
    from alpa_serve.placement_policy.base_policy import ModelData

    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = DummyController()
    register_models(controller)
    placement = place_models(controller)

    model_names, prof_ress = zip(*controller.name2profiling.items())
    model_datas = [ModelData(model_names[i], None, None, None, prof_ress[i])
                   for i in range(len(model_names))]

    # Generate workload
    workload = generate_workload()

    # Load constants
    group_configs, group_models = placement.group_configs, placement.group_models
    name2model_id = {m.name: i for i, m in enumerate(model_datas)}
    num_groups = len(group_configs)
    max_bs = 1
    model_id2group_ids = defaultdict(list)
    for g_id, m_ids in enumerate(group_models):
        for m_id in m_ids:
            model_id2group_ids[m_id].append(g_id)

    # Run load balance
    group_throughputs = [
        1 / max(model_datas[0].profiling_result.para_dict[c].latency[max_bs])
        for c in group_configs]
    group_total_latencies = [
        sum(model_datas[0].profiling_result.para_dict[c].latency[max_bs])
        for c in group_configs]
    group_counters = [0] * num_groups
    group_arrivals = [[] for _ in range(num_groups)]

    request_gids = []
    for i in range(len(workload)):
        model_id = name2model_id[workload.requests[i].model_name]
        group_ids = model_id2group_ids[model_id]

        # Pick the group with least requests
        min_group_id = None
        min_group_value = 1e20
        for g_id in group_ids:
            if group_counters[g_id] / group_throughputs[g_id] < min_group_value:
                min_group_value = group_counters[g_id] / group_throughputs[g_id]
                min_group_id = g_id

        request_gids.append(min_group_id)
        if min_group_id is not None:
            group_arrivals[min_group_id].append(workload.arrivals[i])
            group_counters[min_group_id] += 1

    if debug:
        print(f"group_throughputs: {group_throughputs}, "
              f"group_counters: {group_counters}, "
              f"group_total_latencies: {group_total_latencies}")

    # Compute mean waiting time
    group_waiting_time = []
    for i in range(num_groups):
        arrivals = np.array(group_arrivals[i])
        if len(arrivals) > 1:
            intervals = arrivals[1:] - arrivals[:-1]
            rate = 1 / np.mean(intervals)
            cv = np.std(intervals) * rate
        else:
            rate = 0
            cv = 0

        if debug:
            print(f"rate: {rate:.2f}, cv: {cv:.2f}")

        waiting_time = kingman_formula(rate, cv, group_throughputs[i])
        group_waiting_time.append(waiting_time)

    if debug:
        print(f"group_waiting_time: {group_waiting_time}")

    # Compute e2e latency
    np.random.seed(seed)
    start = workload.arrivals
    finish = []
    good = []
    for i in range(len(workload)):
        m_id = name2model_id[workload.requests[i].model_name]
        g_id = request_gids[i]

        if g_id is None:
            finish.append(None)
            good.append(False)
        else:
            e2e_latency = (np.random.exponential(group_waiting_time[g_id]) +
                group_total_latencies[g_id])
            finish.append(start[i] + e2e_latency)
            good.append(e2e_latency <= workload.requests[i].slo)

    # Compute stats
    stats = workload.compute_stats(start, finish, good, warmup,
                                   compute_tail_latency=False)
    return stats, placement
