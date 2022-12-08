"""
The serving controller.

This file simulates `alpa_serve/controller.py`.
"""
import asyncio
import math
from collections import defaultdict
import dataclasses
from functools import partial
from itertools import cycle
import time
from typing import Callable, List, Dict, Optional, Tuple

import numpy as np
import numba

from alpa_serve.controller import CreateInfo, ModelInfo, GroupInfo, build_logger
from alpa_serve.profiling import ProfilingResult
from alpa_serve.simulator.cluster import VirtualMesh
from alpa_serve.simulator.event_loop import (timed_coroutine, clock,
    main_loop, sleep, run_event_loop)
from alpa_serve.simulator.util import install_remote_methods, async_to_sync
from alpa_serve.simulator.workload import (Workload, StatsResult,
    PerDeviceStatsResult, PerModelStatsResult, DEFAULT_WARMUP)
from alpa_serve.util import ServingCase, inf, eps, to_str_round


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
        self.alpa_overhead = cycle(
            np.random.normal(loc=0.005, scale=0.0005, size=(1024,)))

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

        ret = await self.replicas[name].handle_request(request,
            delay=next(self.alpa_overhead))
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
        self.dispatch_overhead = cycle(
            np.random.normal(loc=0.0025, scale=0.0005, size=(1024,)))

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
            CreateInfo(model_def, init_args, init_kwargs), [], 0)

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

        self.logger.debug(f"Create replica of {name} on group {group_id}")
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
            delay=next(self.dispatch_overhead))
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

    @timed_coroutine
    async def submit_one(self, request, idx, start, finish, good, http_overhead):
        start[idx] = clock()
        request.submit_time = start[idx]
        res = await self.controller.handle_request(request, delay=http_overhead)
        finish[idx] = clock()
        e2e_latency = finish[idx] - start[idx]
        good[idx] = e2e_latency <= request.slo and res is not None

        if self.debug:
            tstamps = to_str_round({x: (y - request.submit_time) * 1e3 for x, y in request.time_stamp.items()}, 2)
            print(f"idx: {idx} ts: {tstamps} e2e latency: {e2e_latency*1e3:.2f} ms", flush=True)

    async def submit_workload(self, workload: Workload):
        num_requests = len(workload)
        start, finish, good = (np.zeros(num_requests),
            np.zeros(num_requests), np.zeros(num_requests, dtype=np.bool))
        self.res_dict[workload] = (start, finish, good)

        http_overheads = np.abs(np.random.normal(
            loc=0.0025, scale=0.0005, size=(num_requests,)))
        for i in range(len(workload)):
            self.submit_one(workload.requests[i], i, start, finish, good,
                            http_overheads[i], tstamp=workload.arrivals[i])

        await main_loop()

    def compute_stats(self, workload: Workload, warmup: float):
        start, finish, good = self.res_dict[workload]
        return workload.compute_stats(start, finish, good, warmup)


async def run_workload(client, workload, warmup):
    await client.submit_workload(workload)
    return client.compute_stats(workload, warmup=warmup)


def simulate_one_case(case: ServingCase, warmup=DEFAULT_WARMUP, debug=False):
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
    stats.group_num_requests = tuple(
        x.num_total_requests for x in controller.group_info.values())
    return stats, placement


class DummyController:
    """A dummy controller used for approximation."""

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

    def sync(self):
        pass


def approximate_one_case(case: ServingCase,
                         seed: int = 0,
                         warmup: int = DEFAULT_WARMUP,
                         debug: bool = False,
                         fast_stats: bool = False):
    """A fast simulator that only simulates one stage for a pipeline."""
    tic = time.time()
    register_models, generate_workload, place_models = case

    workload = generate_workload()

    if workload.enable_simulator_cache and workload.cached_data:
        model_ids, slos, model_names, prof_ress = workload.cached_data
        placement = place_models(None)
    else:
        # Launch the controller
        controller = DummyController()
        register_models(controller)
        placement = place_models(controller)
        # Note: assume the model registration order is the same as the model id order in group_models
        model_names, prof_ress = zip(*controller.name2profiling.items())

        name2model_id = {m: i for i, m in enumerate(model_names)}
        model_ids = np.array([name2model_id.get(r.model_name, -1) for r in workload.requests], dtype=np.int32)
        slos = np.array([r.slo for r in workload.requests], dtype=np.float32)

        if workload.enable_simulator_cache:
            workload.cached_data = (model_ids, slos, model_names, prof_ress)

    # Load constants
    group_configs, group_models = placement.group_configs, placement.group_models

    num_groups = len(group_configs)
    num_models = len(model_names)
    num_requests = len(workload)
    num_replicas = [0] * num_models
    m_id2g_id = np.full((num_models, num_groups), -1, dtype=np.int32)
    for g_id, m_ids in enumerate(group_models):
        for m_id in m_ids:
            m_id2g_id[m_id][num_replicas[m_id]] = g_id
            num_replicas[m_id] += 1

    max_bs = 1
    group_max_latency = np.empty((num_models, num_groups), dtype=np.float32)
    group_sum_latency = np.empty((num_models, num_groups), dtype=np.float32)
    for m_id in range(num_models):
        for g_id in range(num_groups):
            value = prof_ress[m_id].para_dict.get(group_configs[g_id], None)
            if value:
                group_max_latency[m_id][g_id] = max(value.latency[max_bs])
                group_sum_latency[m_id][g_id] = sum(value.latency[max_bs])
            else:
                group_max_latency[m_id][g_id] = group_sum_latency[m_id][g_id] = inf

    # Simulate
    start = workload.arrivals
    finish = np.empty(num_requests, dtype=np.float32)
    good = np.empty(num_requests, dtype=bool)
    tstamps = workload.arrivals

    (model_num_requests, model_num_good_requests,
     group_num_requests, group_num_good_requests) = simulate_requests(
        finish, good, tstamps, model_ids, slos, m_id2g_id,
        group_max_latency, group_sum_latency, num_requests)

    if fast_stats:
        # Note: no warmup
        interval = start[-1] - start[0]
        per_model_stats = [PerModelStatsResult(
            model_names[i], model_num_requests[i],
            model_num_good_requests[i] / (model_num_requests[i] + eps),
            model_num_requests[i] / interval,
            0, 0, 0, 0) for i in range(num_models)]
        stats = StatsResult(per_model_stats, tuple(group_num_requests),
                            np.mean(good), np.mean(finish - start),
                            len(start), len(start) / interval)
    else:
        stats = workload.compute_stats(start, finish, good, warmup)
        stats.group_num_requests = group_num_requests
    return stats, placement


@numba.jit(nopython=True)
def simulate_requests(finish, good, tstamps, model_ids, slos, m_id2g_id,
                      group_max_latency, group_sum_latency, num_requests):
    num_models = len(group_max_latency)
    num_groups = len(group_max_latency[0])

    group_clocks = np.zeros(num_groups, dtype=np.float32)
    group_num_requests = np.zeros(num_groups, dtype=np.int32)
    group_num_good_requests = np.zeros(num_groups, dtype=np.int32)
    model_num_requests = np.zeros(num_models, dtype=np.int32)
    model_num_good_requests = np.zeros(num_models, dtype=np.int32)
    fixed_overhead = 0.011

    for i in range(num_requests):
        tstamp, m_id, slo = tstamps[i], model_ids[i], slos[i]

        if m_id < 0:
            finish[i] = tstamp
            good[i] = False
            continue

        # Select group id
        g_id = -1
        min_group_clock = inf
        for j in m_id2g_id[m_id]:
            if j < 0:
                break
            if group_clocks[j] < min_group_clock:
                min_group_clock = group_clocks[j]
                g_id = j

        if g_id < 0:
            finish[i] = tstamp
            good[i] = False
            continue

        start_time = max(group_clocks[g_id], tstamp)
        finish_time = start_time + group_sum_latency[m_id][g_id] + fixed_overhead
        group_num_requests[g_id] += 1
        model_num_requests[m_id] += 1

        if finish_time - tstamp <= slo:
            finish[i] = finish_time
            good[i] = True
            group_clocks[g_id] = start_time + group_max_latency[m_id][g_id]
            group_num_good_requests[g_id] += 1
            model_num_good_requests[m_id] += 1
        else:
            finish[i] = tstamp
            good[i] = False

    return (model_num_requests, model_num_good_requests,
            group_num_requests, group_num_good_requests)
