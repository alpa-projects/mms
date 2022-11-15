"""The baseclass of model placement policy"""
import dataclasses
import time
from typing import List

import numpy as np

from alpa_serve.profiling import ProfilingResult


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

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.group_configs = None
        self.group_models = None
        self.debug_info = None

    def place_models(self, controller,
                     model_datas: List[ModelData], cluster_env: ClusterEnv):
        tic = time.time()
        (self.group_configs, self.group_models, self.debug_info
         ) = self.solve_placement(model_datas, cluster_env)
        solver_time = time.time() - tic

        assert len(self.group_configs) == len(self.group_models)
        num_groups = len(self.group_configs)

        # Create mesh group manager
        for g_id in range(num_groups):
            num_devices = np.prod(self.group_configs[g_id])
            num_devices_per_node = cluster_env.num_devices_per_node
            pp_size = self.group_configs[g_id].pp

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
            for m_id in self.group_models[g_id]:
                name = model_datas[m_id].name
                controller.create_replica.remote(name, g_id, [self.group_configs[g_id]])
        controller.sync()

        if self.verbose:
            print(f"group configs: {self.group_configs}")
            print(f"group models: {self.group_models}")
            print(f"debug info: {self.debug_info}")
            print(f"solver time: {solver_time:.2f}")

    def __str__(self):
        group_strs = [f"({config}, {models})" for config, models
                      in zip(self.group_configs, self.group_models)]
        return f"{self.__class__.__name__}([" + ", ".join(group_strs) + "])"
