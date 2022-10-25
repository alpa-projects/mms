"""
The serving controller.

This file simulates `alpa_serve/controler.py`.
"""
import asyncio
import dataclasses
from functools import partial
import math
from typing import Callable, List, Dict, Optional, Tuple, Any, Union

import numpy as np

from alpa_serve.controller import CreateInfo, ModelInfo, GroupInfo
from alpa_serve.placement_policy import ParallelConfig
from alpa_serve.simulator.cluster import VirtualMesh
from alpa_serve.simulator.event_loop import (timed_coroutine, wait_stream, sleep,
    clock, init_event_loop, main_task)
from alpa_serve.simulator.executable import Executable, ProfilingResult
from alpa_serve.simulator.workload import Workload
from alpa_serve.simulator.util import install_remote_methods


class GroupManager:
    def __init__(self, virtual_mesh_shape):
        self.virtual_mesh = VirtualMesh(virtual_mesh_shape)

        # Dict[str -> object]
        self.replicas = {}

    def create_replica(self, name: str, create_info: CreateInfo):
        assert name not in self.replicas

        model_def, args, kwargs = (create_info.model_def, create_info.init_args,
                                   create_info.init_kwargs)
        args = args or []
        kwargs = kwargs or {}
        kwargs["virtual_mesh"] = self.virtual_mesh
        self.replicas[name] = model_def(*args, **kwargs)

    @timed_coroutine
    async def handle_request(self, name: str, request):
        await self.replicas[name].execute(batch_size=1)


class Controller:
    def __init__(self):
        # Dict[str -> ModelInfo]
        self.model_info = {}
        # Dict[int -> GroupInfo]
        self.group_info = {}
        # Dict[int -> ActorHandle]
        self.group_managers = {}

        self.dispatch_overhead = 0

        install_remote_methods(self)

    def create_mesh_group_manager(
            self,
            group_id: int,
            virtual_mesh_shape: Optional[Tuple[int]] = None,
            num_gpus: int = 0):
        assert group_id not in self.group_managers, (
            f"Mesh group {group_id} is already launched")
        self.group_managers[group_id] = GroupManager(virtual_mesh_shape)
        self.group_info[group_id] = GroupInfo(queue_size=0)

    def register_model(self,
                       name: str,
                       model_def: Callable,
                       init_args: Optional[List] = None,
                       init_kwargs: Optional[Dict] = None,
                       override: bool = False):
        if name in self.model_info:
            if override:
                for manager in self.model_info[name].managers:
                    manager.delete_replica(name)
            else:
                raise ValueError(f"Model {name} is already registered")

        self.model_info[name] = ModelInfo(
            CreateInfo(model_def, init_args, init_kwargs), [])

    def create_replica(self,
                       name: str,
                       group_id: int,
                       append_init_args: Optional[List] = None,
                       append_init_kwargs: Optional[Dict] = None):
        assert group_id in self.group_managers, (
            f"Group {group_id} does not exist")
        model_info = self.model_info[name]
        manager = self.group_managers[group_id]
        assert manager not in model_info.managers
        create_info = model_info.create_info.append_init_args(
            append_init_args, append_init_kwargs)

        model_info.managers.append(manager)
        manager.create_replica(name, create_info)

    def select_group_id(self):
        min_id = -1
        min_size = math.inf
        for group_id, group_info in self.group_info.items():
            if group_info.queue_size < min_size:
                min_size = group_info.queue_size
                min_id = group_id
        assert min_id != -1
        return min_id

    @timed_coroutine
    async def handle_request(self, request):
        start = clock()

        name = request.model_name

        assert name in self.model_info, (
            f"Model '{name}' is not registered.")
        model_info = self.model_info[name]
        assert model_info.managers, (
            f"No replica of model '{name}' is created.")

        # dispatch
        group_id = self.select_group_id()
        manager = self.group_managers[group_id]

        self.group_info[group_id].queue_size += 1
        response = await manager.handle_request(name, request,
                                                delay=self.dispatch_overhead)
        self.group_info[group_id].queue_size -= 1

        return response


class Client:
    def __init__(self, controller):
        self.controller = controller

        self.res_dict = dict()

    @timed_coroutine
    async def submit_one(self, request, idx, start, finish):
        start[idx] = clock()
        await self.controller.handle_request(request)
        finish[idx] = clock()

    def submit_workload(self, workload: Workload):
        start, finish = np.zeros(len(workload)), np.zeros(len(workload))
        self.res_dict[workload] = (start, finish)

        for i in range(len(workload)):
            self.submit_one(workload.requests[i], i, start, finish,
                            tstamp=workload.arrivals[i])

    def print_stats(self, workload: Workload, warmup: float):
        start, finish = self.res_dict[workload]
        workload.print_stats(start, finish, warmup)


async def test_main():
    init_event_loop()

    controller = Controller()
    controller.register_model.remote(
        "a", partial(Executable, ProfilingResult.load("alpa/bert-1.3b")))

    group_id = 0
    controller.create_mesh_group_manager.remote(group_id, [1, 2])
    controller.create_replica.remote("a", group_id,
                                     [ParallelConfig(1, 1, 2)])

    w = Workload.gen_poisson("a", 0, 10, 60)
    client = Client(controller)
    client.submit_workload(w)

    await main_task()

    client.print_stats(w, warmup=10)


if __name__ == "__main__":
    asyncio.run(test_main())
