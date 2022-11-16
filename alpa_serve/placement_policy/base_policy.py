"""The baseclass of model placement policy"""
import dataclasses
import time
from typing import List

import numpy as np

from alpa_serve.profiling import ProfilingResult, ParallelConfig


@dataclasses.dataclass
class ModelPlacement:
    group_configs: List[ParallelConfig]
    group_models: List[List[int]]


@dataclasses.dataclass
class ModelData:
    name: str
    slo: float
    rate: float
    cv: float
    profiling_result: ProfilingResult


@dataclasses.dataclass
class ClusterEnv:
    num_devices: int
    mem_budget: float
    num_devices_per_node: int = 8


class BasePlacementPolicy:
    """The baseclass of placement policy"""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    def place_models(self, controller, cluster_env: ClusterEnv,
                     model_datas: List[ModelData]):
        tic = time.time()
        (placement, debug_info) = self.solve_placement(model_datas, cluster_env)
        solver_time = time.time() - tic

        self.place_models_impl(controller, cluster_env, model_datas, placement)

        if self.verbose >= 1:
            print(f"placement solution: {placement}")
            print(f"debug info: {debug_info}")
            print(f"solver time: {solver_time:.2f} s")

        return placement

    def place_models_impl(self, controller,
                          cluster_env: ClusterEnv,
                          model_datas: List[ModelData],
                          placement: ModelPlacement):
        group_configs, group_models = placement.group_configs, placement.group_models
        assert len(group_configs) == len(group_models)
        num_groups = len(group_configs)

        # Create mesh group manager
        for g_id in range(num_groups):
            num_devices = np.prod(group_configs[g_id])
            num_devices_per_node = cluster_env.num_devices_per_node

            if num_devices <= num_devices_per_node:
                virtual_mesh_shape = (1, num_devices)
            else:
                assert num_devices % num_devices_per_node == 0
                virtual_mesh_shape = (num_devices // num_devices_per_node,
                                      num_devices_per_node)

            controller.create_mesh_group_manager.remote(g_id, virtual_mesh_shape)
        controller.sync()

        # Create model replicas
        for g_id in range(num_groups):
            for m_id in group_models[g_id]:
                name = model_datas[m_id].name
                controller.create_replica.remote(name, g_id, [group_configs[g_id]])
        controller.sync()
