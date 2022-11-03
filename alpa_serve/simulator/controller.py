"""
The serving controller.

This file simulates `alpa_serve/controler.py`.
"""
import asyncio
import math
from collections import defaultdict
from functools import partial
from typing import Callable, List, Dict, Optional, Tuple

import numpy as np

from alpa_serve.controller import CreateInfo, ModelInfo, GroupInfo, build_logger
from alpa_serve.simulator.cluster import VirtualMesh
from alpa_serve.simulator.event_loop import timed_coroutine, clock, main_loop, sleep
from alpa_serve.simulator.util import install_remote_methods, async_to_sync
from alpa_serve.simulator.workload import Workload
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

        self.logger = build_logger()

        # Constants
        self.fixed_overhead = 0.005

        self.alpa_overhead = partial(np.random.normal, loc=0.004, scale=0.001)

        # Simulator specific code
        install_remote_methods(self)

    async def create_replica(self, name: str, create_info: CreateInfo):
        assert name not in self.replicas

        model_def, args, kwargs = (create_info.model_def, create_info.init_args,
                                   create_info.init_kwargs)
        args = args or []
        kwargs = kwargs or {}
        kwargs["virtual_mesh"] = self.virtual_mesh
        self.replicas[name] = model_def(*args, **kwargs)

        self.latency_dict[name] = self.replicas[name].get_latency_dict()

    @timed_coroutine
    async def handle_request(self, name: str, request):
        request.time_stamp["b"] = clock()

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

        self.logger = build_logger()

        # Simulator specific code
        np.random.seed(1)
        self.dispatch_overhead = partial(np.random.normal, loc=0.002, scale=0.0015)

        install_remote_methods(self)

        group_manager_init = partial(lambda: None)
        group_manager_init.remote = partial(GroupManager)
        self.group_manager_class = partial(lambda: None)
        self.group_manager_class.options = lambda *args, **kwargs: group_manager_init

    @async_to_sync
    async def create_mesh_group_manager(
            self,
            group_id: int,
            virtual_mesh_shape: Optional[Tuple[int]] = None,
            num_gpus: int = 0):
        assert group_id not in self.group_info, (
            f"Mesh group {group_id} is already launched")
        self.logger.info(f"Create mesh group manager {group_id} with "
                         f"shape={virtual_mesh_shape}")
        manager = (self.group_manager_class.options(
            name=f"mesh_group_manager_{group_id}",
            num_gpus=num_gpus).remote(virtual_mesh_shape))
        self.group_info[group_id] = GroupInfo(manager=manager, queue_size=0)

    @async_to_sync
    async def register_model(self,
                             name: str,
                             model_def: Callable,
                             init_args: Optional[List] = None,
                             init_kwargs: Optional[Dict] = None,
                             override: bool = False):
        async with self.manager_lock[name]:
            if name in self.model_info:
                if override:
                    for group_id in self.model_info[name].group_ids:
                        await self.group_info[group_id
                            ].manager.delete_replica.remote(name)
                else:
                    raise ValueError(f"Model {name} is already registered")

            self.model_info[name] = ModelInfo(
                CreateInfo(model_def, init_args, init_kwargs), [])

    @async_to_sync
    async def create_replica(self,
                             name: str,
                             group_id: int,
                             append_init_args: Optional[List] = None,
                             append_init_kwargs: Optional[Dict] = None):
        async with self.manager_lock[name]:
            assert group_id in self.group_info, (
                f"Group {group_id} does not exist")
            model_info = self.model_info[name]
            manager = self.group_info[group_id].manager
            assert group_id not in model_info.group_ids
            create_info = model_info.create_info.append_init_args(
                append_init_args, append_init_kwargs)

            self.logger.info(f"Create replica of {name} on group {group_id}")
            model_info.group_ids.append(group_id)
        await manager.create_replica.remote(name, create_info)

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
        assert model_info.group_ids, (
            f"No replica of model '{name}' is created.")

        # Dispatch
        group_id = self.select_group_id(model_info.group_ids)
        manager = self.group_info[group_id].manager

        self.group_info[group_id].queue_size += 1
        response = await manager.handle_request.remote(name, request,
            delay=abs(self.dispatch_overhead()))
        self.group_info[group_id].queue_size -= 1

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
        t = clock()
        start[idx] = t
        request.submit_time = t
        res = await self.controller.handle_request(request, delay=self.http_overhead())
        t = clock()
        finish[idx] = t
        good[idx] = (res is not None and
                     t <= request.submit_time + request.slo)

        if self.debug:
            e2e_latency = finish[idx] - start[idx]
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
