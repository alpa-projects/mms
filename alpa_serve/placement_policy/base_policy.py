"""The baseclass of model placement policy"""
import dataclasses
from functools import partial
import logging
import time
from typing import List

import numpy as np
import ray

from alpa_serve.profiling import ProfilingResult, ParallelConfig
from alpa_serve.simulator.controller import simulate_one_case, approximate_one_case
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload
from alpa_serve.util import ServingCase


@dataclasses.dataclass
class ModelPlacement:
    group_configs: List[ParallelConfig]
    group_models: List[List[int]]

    def add_model(self, group_id: int, model_id: int):
        group_models = list(self.group_models)
        group_models[group_id] = list(group_models[group_id])
        group_models[group_id].append(model_id)
        return ModelPlacement(self.group_configs, group_models)

    def normalize(self):
        group_models = []
        for i in range(len(self.group_models)):
            group_models.append(tuple(sorted(self.group_models[i])))
        group_models = tuple(group_models)
        return ModelPlacement(self.group_configs, group_models)


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
                     model_datas: List[ModelData], train_workload: Workload = None):
        tic = time.time()
        (placement, debug_info) = self.solve_placement(model_datas, cluster_env, train_workload)
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


def get_scores(sols: List[ModelPlacement], model_datas: List[ModelData],
               cluster_env: ClusterEnv, workload: Workload, approximate: bool,
               parallel: bool):
    """Get goodput of placements through simulation."""
    if parallel:
        get_score_one_sol_ = ray.remote(num_cpus=1)(get_score_one_sol).remote

        model_datas = ray.put(model_datas)
        cluster_env = ray.put(cluster_env)
        workload = ray.put(workload)

        scores = [get_score_one_sol_(sol, model_datas, cluster_env,
                                     workload, approximate)
                  for sol in sols]
        scores = ray.get(scores)
    else:
        scores = [get_score_one_sol(sol, model_datas, cluster_env,
                                    workload, approximate)
                  for sol in sols]
    return scores


def get_score_one_sol(sol: ModelPlacement,
                      model_datas: List[ModelData],
                      cluster_env: ClusterEnv,
                      workload: Workload,
                      approximate: bool):
    def register_models(controller):
        for i, data in enumerate(model_datas):
            controller.register_model.remote(
                data.name, partial(Executable, data.profiling_result))
            if not approximate:
                controller.logger.setLevel(logging.ERROR)

    def generate_workload(start=0):
        return workload

    def place_models(controller):
        if approximate:
            return sol
        else:
            base_policy = BasePlacementPolicy()
            base_policy.place_models_impl(controller, cluster_env, model_datas, sol)

    serving_case = ServingCase(register_models, generate_workload, place_models)
    run_func = approximate_one_case if approximate else simulate_one_case
    stats, _ = run_func(serving_case)
    num_replicas = sum(len(x) for x in sol.group_models)
    return stats.goodput - stats.latency_mean / 10000 + num_replicas / 1000000
