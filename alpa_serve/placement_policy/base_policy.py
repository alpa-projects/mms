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
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.util import ServingCase, inf, to_str_round


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
        group_models = tuple(sorted(tuple(sorted(x)) for x in self.group_models))
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


class PlacementEvaluator:
    """Evaluate the scores of model placements via the simulator or other
    approximations."""

    def __init__(self,
                 model_datas: List[ModelData],
                 cluster_env: ClusterEnv,
                 workload: Workload,
                 method: str,
                 parallel: bool):
        self.parallel = parallel

        if parallel:
            self.model_datas = ray.put(model_datas)
            self.cluster_env = ray.put(cluster_env)
            self.workload = ray.put(workload)
            self.method = ray.put(method)
            self.get_score_one_sol = ray.remote(num_cpus=1)(
                self.get_goodput_simulation).remote
            self.get_stats_one_sol = ray.remote(num_cpus=1)(
                self.get_stats_simulation).remote
        else:
            self.model_datas = model_datas
            self.cluster_env = cluster_env
            self.workload = workload
            workload.enable_simulator_cache = True
            self.method = method
            self.get_score_one_sol = self.get_goodput_simulation
            self.get_stats_one_sol = self.get_stats_simulation

    def get_scores(self, sols: List[ModelPlacement]):
        scores = [self.get_score_one_sol(sol, self.model_datas,
            self.cluster_env, self.workload, self.method) for sol in sols]

        if self.parallel:
            scores = ray.get(scores)
        return scores

    def get_stats(self, sols: List[ModelPlacement]):
        stats = [self.get_stats_one_sol(sol, self.model_datas,
            self.cluster_env, self.workload, self.method) for sol in sols]

        if self.parallel:
            stats = ray.get(stats)
        return stats

    @staticmethod
    def get_goodput_simulation(sol: ModelPlacement,
                               model_datas: List[ModelData],
                               cluster_env: ClusterEnv,
                               workload: Workload,
                               method: str):
        if method == "fast_simulator":
            fast_simulator = True
        else:
            fast_simulator = False

        def register_models(controller):
            for i, data in enumerate(model_datas):
                controller.register_model.remote(
                    data.name, partial(Executable, data.profiling_result))

                if not fast_simulator:
                    controller.logger.setLevel(logging.ERROR)

        def generate_workload(start=0):
            return workload

        def place_models(controller):
            if fast_simulator:
                return sol
            else:
                base_policy = BasePlacementPolicy()
                base_policy.place_models_impl(controller, cluster_env, model_datas, sol)

        serving_case = ServingCase(register_models, generate_workload, place_models)
        if fast_simulator:
            stats, _ = approximate_one_case(serving_case, compute_per_model_stats=False)
        else:
            stats, _ = simulate_one_case(serving_case)
        num_replicas = sum(len(x) for x in sol.group_models)
        return stats.goodput - stats.latency_mean / 10000 + num_replicas / 1000000

    @staticmethod
    def get_stats_simulation(sol: ModelPlacement,
                             model_datas: List[ModelData],
                             cluster_env: ClusterEnv,
                             workload: Workload,
                             method: str):
        if method == "fast_simulator":
            fast_simulator = True
        else:
            fast_simulator = False

        def register_models(controller):
            for i, data in enumerate(model_datas):
                controller.register_model.remote(
                    data.name, partial(Executable, data.profiling_result))

                if not fast_simulator:
                    controller.logger.setLevel(logging.ERROR)

        def generate_workload(start=0):
            return workload

        def place_models(controller):
            if fast_simulator:
                return sol
            else:
                base_policy = BasePlacementPolicy()
                base_policy.place_models_impl(controller, cluster_env, model_datas, sol)

        serving_case = ServingCase(register_models, generate_workload, place_models)

        if fast_simulator:
            stats, _ = approximate_one_case(serving_case)
        else:
            stats, _ = simulate_one_case(serving_case)

        name2id = {m.name: m_id for m_id, m in enumerate(model_datas)}
        model_goodput = np.empty(len(name2id), dtype=np.float32)
        for m in stats.per_model_stats:
            model_goodput[name2id[m.name]] = m.goodput

        return (stats.goodput, model_goodput, stats.group_num_requests)


def gen_train_workload(model_datas: List[ModelData],
                       seed: int = 1234,
                       simulation_min_duration: float = 100,
                       simulation_min_samples: int = 30000):
    """Generate a training workload for search."""
    total_rate = sum(d.rate for d in model_datas)
    duration = max(simulation_min_duration, simulation_min_samples / total_rate)

    ws = []
    for i, data in enumerate(model_datas):
        ws.append(GammaProcess(data.rate, data.cv).generate_workload(
            data.name, 0, duration=duration,
            slo=data.slo, seed=seed + i))
    train_workload = Workload.merge(*ws)
    return train_workload


def replica_placement_fast_greedy(init_sol: ModelPlacement,
                                  model_datas: List[ModelData],
                                  cluster_env: ClusterEnv,
                                  workload: Workload,
                                  evaluator: PlacementEvaluator,
                                  verbose: int):
    """Use a fast greedy heuristic to place replicas on groups."""
    tic = time.time()

    if evaluator is None:
        evaluator = PlacementEvaluator(model_datas, cluster_env, workload,
            "fast_simulator", False)

    # Load constants
    num_models = len(model_datas)
    num_groups = len(init_sol.group_configs)
    mem_budget = cluster_env.mem_budget
    group_configs = init_sol.group_configs

    weight_mem = {}  # Dict[parallel_config -> [model_idx -> weight_mem]]
    for parallel_config in init_sol.group_configs:
        weight_mem[parallel_config] = [
            max(x.profiling_result.para_dict[parallel_config].weight_mem)
            if parallel_config in x.profiling_result.para_dict
            else inf
            for x in model_datas]

    rates = [m.rate for m in model_datas]

    # Greedy placement
    sol = init_sol
    it = 0

    while True:
        stats = evaluator.get_stats([sol])[0]
        overall_goodput, goodputs, group_num_requests = stats

        # Find the most unserved model and the most available group
        model_num_unserved = [
            (rate * (1 - goodput)) for rate, goodput in zip(rates, goodputs)]
        model_ids = np.argsort(model_num_unserved)[::-1]
        group_ids = np.argsort(group_num_requests)
        group_mem = [
            sum(weight_mem[c][m_id] for m_id in group_ms)
            for c, group_ms in zip(sol.group_configs, sol.group_models)
        ]

        found = False
        for g_id in group_ids:
            c = sol.group_configs[g_id]
            for m_id in model_ids:
                if (m_id not in sol.group_models[g_id] and
                    weight_mem[c][m_id] + group_mem[g_id] <= mem_budget):
                    found = True
                    break

            if found:
                break
        if not found:
            break

        sol = sol.add_model(g_id, m_id).normalize()

        if verbose >= 2:
            print(f"iter: {it}, score: {overall_goodput:.4f}, "
                  f"elapsed: {time.time() - tic:.2f}, "
                  f"best placement: {sol}, ")
        it += 1

    return sol


def replica_placement_beam_search(init_sol: ModelPlacement,
                                  model_datas: List[ModelData],
                                  cluster_env: ClusterEnv,
                                  workload: Workload,
                                  evaluator: PlacementEvaluator,
                                  beam_size: int,
                                  verbose: int):
    """Use beam search to place replicas on groups."""
    tic = time.time()

    if evaluator is None:
        evaluator = PlacementEvaluator(model_datas, cluster_env, workload,
            "fast_simulator", False)

    # Load constants
    num_models = len(model_datas)
    num_groups = len(init_sol.group_configs)
    mem_budget = cluster_env.mem_budget
    group_configs = init_sol.group_configs

    weight_mem = {}  # Dict[parallel_config -> [model_idx -> weight_mem]]
    for parallel_config in init_sol.group_configs:
        weight_mem[parallel_config] = [
            max(x.profiling_result.para_dict[parallel_config].weight_mem)
            if parallel_config in x.profiling_result.para_dict
            else inf
            for x in model_datas]

    # Beam search
    beam = [init_sol]
    it = 0

    best_score = -1
    best_sol = init_sol
    visited = set()

    while True:
        # Expand one layer
        next_sols = []
        for sol in beam:
            group_mem = [
                sum(weight_mem[c][m_id] for m_id in group_ms)
                for c, group_ms in zip(sol.group_configs, sol.group_models)
            ]
            for g_id in range(num_groups):
                c = sol.group_configs[g_id]
                for m_id in range(num_models):
                    if (weight_mem[c][m_id] + group_mem[g_id] < mem_budget and
                        m_id not in sol.group_models[g_id]):
                        next_sol = sol.add_model(g_id, m_id).normalize()

                        if next_sol.group_models not in visited:
                            visited.add(next_sol.group_models)
                            next_sols.append(next_sol)
        if not next_sols:
            break

        # Pick the new top-k
        next_scores = evaluator.get_scores(next_sols)
        next_indices = np.argsort(next_scores)[::-1][:beam_size]

        beam = []
        for idx in next_indices:
            beam.append(next_sols[idx])
            if next_scores[idx] > best_score:
                best_score = next_scores[idx]
                best_sol = next_sols[idx]

        if verbose >= 1:
            print(f"iter: {it}, best score: {best_score:.4f}, "
                  f"iter score: {next_scores[next_indices[0]]:.4f}, "
                  f"iter #sol: {len(next_sols)}, "
                  f"elapsed: {time.time() - tic:.2f}, "
                  f"best placement: {best_sol}, ")
        it += 1

    return best_sol


def swap_two_models(sol: ModelPlacement):
    group_models = sol.group_models
    g_id_1 = np.random.choice(len(group_models))
    g_id_2 = np.random.choice(len(group_models))
    m_id_1 = np.random.choice(len(group_models[g_id_1]))
    m_id_2 = np.random.choice(len(group_models[g_id_2]))
    if (group_models[g_id_1][m_id_1] in group_models[g_id_2] or
        group_models[g_id_2][m_id_2] in group_models[g_id_1]):
        return sol
    group_models = [list(x) for x in sol.group_models]
    group_models[g_id_1][m_id_1], group_models[g_id_2][m_id_2] = (
        group_models[g_id_2][m_id_2], group_models[g_id_1][m_id_1])
    return ModelPlacement(sol.group_configs, group_models)


def mutate_one_model(sol: ModelPlacement, num_models: int):
    group_models = sol.group_models
    g_id = np.random.choice(len(group_models))
    new_model_id = np.random.choice(num_models)
    if new_model_id in group_models[g_id]:
        return sol
    m_id_1 = np.random.choice(len(group_models[g_id]))
    group_models = [list(x) for x in sol.group_models]
    group_models[g_id][m_id_1] = new_model_id
    return ModelPlacement(sol.group_configs, group_models)


def evolutionary_search(init_sols: List[ModelPlacement],
                        model_datas: List[ModelData],
                        evaluator: PlacementEvaluator,
                        verbose: int):
    tic = time.time()

    # Constants
    pop_size = 1024
    mutation_prob = 0.04
    num_iter = 200
    num_models = len(model_datas)

    # Search status
    best_score = -1
    best_sol = None
    it = 0
    visited = set()

    # Iterative search
    cur_sols = init_sols
    while it < num_iter:
        scores = np.asarray(evaluator.get_scores(cur_sols), dtype=np.float32)
        weights = scores / np.sum(scores)

        tmp_best_idx = np.argmax(scores)
        if scores[tmp_best_idx] > best_score:
            best_score = scores[tmp_best_idx]
            best_sol = cur_sols[tmp_best_idx]

        next_sols = []
        while len(next_sols) < pop_size:
            sol = cur_sols[np.random.choice(len(scores), p=weights)]

            # random mutation
            group_models = [list(x) for x in sol.group_models]
            for g_id in range(len(sol.group_models)):
                for m_id in range(len(sol.group_models[g_id])):
                    if np.random.uniform() < mutation_prob:
                        new_m_id = np.random.choice(num_models)
                        if new_m_id not in sol.group_models[g_id]:
                            group_models[g_id][m_id] = new_m_id

            new_sol = ModelPlacement(sol.group_configs, group_models)
            next_sols.append(new_sol)
            visited.add(new_sol.normalize().group_models)

        if verbose >= 1:
            print(f"iter: {it}, best score: {best_score:.4f}, "
                  f"iter score: {scores[tmp_best_idx]:.4f}, "
                  f"iter #sol: {len(scores)}, "
                  f"visited #sol: {len(visited)}, "
                  f"elapsed: {time.time() - tic:.2f}, "
                  f"best placement: {best_sol}, ")

        it += 1
        cur_sols = next_sols + [best_sol]

    return best_sol
