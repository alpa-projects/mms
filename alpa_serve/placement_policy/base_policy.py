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
        indices = list(range(len(self.group_configs)))
        group_models = tuple(tuple(sorted(x)) for x in self.group_models)
        indices.sort(key=lambda i: group_models[i])
        group_configs = tuple(self.group_configs[i] for i in indices)
        group_models = tuple(group_models[i] for i in indices)
        return ModelPlacement(group_configs, group_models)

    def copy(self):
        group_models = list(list(x) for x in self.group_models)
        return ModelPlacement(list(self.group_configs), group_models)

    def verify(self, model_datas, cluster_env):
        weight_mem = {}  # Dict[parallel_config -> [model_idx -> weight_mem]]
        for parallel_config in self.group_configs:
            weight_mem[parallel_config] = [
                max(x.profiling_result.para_dict[parallel_config].weight_mem)
                if parallel_config in x.profiling_result.para_dict
                else inf
                for x in model_datas]

        group_mem = [
            sum(weight_mem[c][m_id] for m_id in group_ms)
            for c, group_ms in zip(self.group_configs, self.group_models)
        ]
        assert all(mem <= cluster_env.mem_budget for mem in group_mem)
        assert all(len(set(ms)) == len(ms) for ms in self.group_models)


@dataclasses.dataclass
class ModelPlacementWithReplacement:
    start_times: List[float]
    placements: List[ModelPlacement]

    def verify(self, model_datas, cluster_env):
        for p in self.placements:
            p.verify(model_datas, cluster_env)

    def __str__(self):
        return f"ModelPlacementWithReplacement(num_segments={len(self.placements)})"


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

        placement.verify(model_datas, cluster_env)
        return placement

    def place_models_impl(self, controller,
                          cluster_env: ClusterEnv,
                          model_datas: List[ModelData],
                          placement: ModelPlacement):
        if isinstance(placement, ModelPlacementWithReplacement):
            return

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

        workload.cached_data = None

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
            stats, _ = approximate_one_case(serving_case, fast_stats=True)
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
            stats, _ = approximate_one_case(serving_case, fast_stats=True)
        else:
            stats, _ = simulate_one_case(serving_case)
        model_goodput = [x.goodput for x in stats.per_model_stats]
        return (stats.goodput, model_goodput, stats.group_num_requests, stats)


def gen_train_workload(model_datas: List[ModelData],
                       seed: int = 0,
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


def replica_placement_round_robin(init_sol: ModelPlacement,
                                  model_datas: List[ModelData],
                                  cluster_env: ClusterEnv,
                                  workload: Workload,
                                  verbose: int):
    """Use round robin to place replicas on groups."""

    assert len(init_sol.group_configs) == len(init_sol.group_models)

    # Load constants
    num_models = len(model_datas)
    num_groups = len(init_sol.group_configs)
    mem_budget = cluster_env.mem_budget
    num_devices = cluster_env.num_devices

    weight_mem = {}  # Dict[parallel_config -> [model_idx -> weight_mem]]
    for parallel_config in init_sol.group_configs:
        weight_mem[parallel_config] = [
            max(x.profiling_result.para_dict[parallel_config].weight_mem)
            if parallel_config in x.profiling_result.para_dict
            else inf
            for x in model_datas]

    sol = init_sol
    group_mem = [
        sum(weight_mem[c][m_id] for m_id in group_ms)
        for c, group_ms in zip(sol.group_configs, sol.group_models)
    ]
    group_id = 0
    found = True
    while found:
        found = False
        for model_id in range(num_models):
            c = sol.group_configs[group_id]
            if (model_id not in sol.group_models[group_id] and
                weight_mem[c][model_id] + group_mem[group_id] <= mem_budget):
                found = True
                group_mem[group_id] += weight_mem[c][model_id]
                sol = sol.add_model(group_id, model_id)
                sol.verify(model_datas, cluster_env)
                group_id = (group_id + 1) % num_groups

    return sol


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

    # Greedy placement
    sol = init_sol
    it = 0

    while True:
        stats = evaluator.get_stats([sol])[0]
        overall_goodput, goodputs, group_num_requests, fullstats = stats

        # Find the most unserved model and the most available group
        model_num_unserved = [
            (s.num_requests * (1 - goodput))
            for s, goodput in zip(fullstats.per_model_stats, goodputs)]
        #model_num_unserved = [
        #    (x.rate * (1 - goodput))
        #    for x, goodput in zip(model_datas, goodputs)]
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


def replica_placement_on_last_group(init_sol: ModelPlacement,
                                    model_datas: List[ModelData],
                                    cluster_env: ClusterEnv,
                                    workload: Workload,
                                    evaluator: PlacementEvaluator,
                                    beam_size: int,
                                    verbose: int):
    """Use beam search to place replicas on the last group."""
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
            g_id = num_groups - 1
            c = sol.group_configs[g_id]
            for m_id in range(num_models):
                if (weight_mem[c][m_id] + group_mem[g_id] < mem_budget and
                    m_id not in sol.group_models[g_id]):
                    next_sol = sol.add_model(g_id, m_id)
                    next_sol_norm = next_sol.normalize()

                    if next_sol_norm.group_models not in visited:
                        visited.add(next_sol_norm.group_models)
                        next_sols.append(next_sol)
 
                        # swap model m_id with a model in previous groups
                        for m_id_1 in range(len(next_sol.group_models[-1])):
                            if next_sol.group_models[-1][m_id_1] == m_id:
                                break
                        for g_id_2 in range(num_groups - 1):
                            for m_id_2 in range(len(next_sol.group_models[g_id_2])):
                                if (next_sol.group_models[g_id][m_id_1]
                                        in next_sol.group_models[g_id_2] or
                                    next_sol.group_models[g_id_2][m_id_2]
                                        in next_sol.group_models[g_id]):
                                    continue
                                group_models = [list(x) for x in next_sol.group_models]
                                group_models[g_id][m_id_1], group_models[g_id_2][m_id_2] = (
                                group_models[g_id_2][m_id_2], group_models[g_id][m_id_1])
                                swap_sol = ModelPlacement(next_sol.group_configs, group_models)
                                swap_sol_norm = swap_sol.normalize()
                                if swap_sol_norm.group_models not in visited:
                                    visited.add(swap_sol_norm.group_models)
                                    next_sols.append(swap_sol)

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


def swap_two_models_from_two_groups(sol: ModelPlacement, g_id_1, g_id_2):
    group_models = sol.group_models
    m_id_1 = np.random.choice(len(group_models[g_id_1]))
    m_id_2 = np.random.choice(len(group_models[g_id_2]))
    if (group_models[g_id_1][m_id_1] in group_models[g_id_2] or
        group_models[g_id_2][m_id_2] in group_models[g_id_1]):
        return False, sol
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
                        cluster_env: ClusterEnv,
                        evaluator: PlacementEvaluator,
                        num_iter: int,
                        verbose: int):
    tic = time.time()

    # Constants
    pop_size = 1024
    mutate_one_model_prob = 0.05
    merge_group_prob = 0.08
    split_group_prob = 0.08
    num_models = len(model_datas)

    mem_budget = cluster_env.mem_budget
    weight_mem = {}  # Dict[parallel_config -> [model_idx -> weight_mem]]
    for m_id, x in enumerate(model_datas):
        for c in x.profiling_result.para_dict:
            if c not in weight_mem:
                weight_mem[c] = [inf] * len(model_datas)
            weight_mem[c][m_id] = max(x.profiling_result.para_dict[c].weight_mem)

    # Search status
    best_score = -1
    best_sol = None
    it = 0
    visited = set()

    # Iterative search
    cur_sols = init_sols
    while it < num_iter:
        stats = evaluator.get_stats(cur_sols)
        scores = np.array([x[0] for x in stats])
        weights = scores / np.sum(scores)
        model_num_unserved_list = [None] * len(stats)

        tmp_best_idx = np.argmax(scores)
        if scores[tmp_best_idx] > best_score:
            best_score = scores[tmp_best_idx]
            best_sol = cur_sols[tmp_best_idx]

        next_sols = []
        while len(next_sols) < pop_size:
            idx = np.random.choice(len(scores), p=weights)
            sol = cur_sols[idx]
            goodputs = stats[idx][1]
            fullstats = stats[idx][3]

            if model_num_unserved_list[idx] is not None:
                model_num_unserved = model_num_unserved_list[idx]
            else:
                model_num_unserved = [
                    (s.num_requests * (1 - goodput))
                    for s, goodput in zip(fullstats.per_model_stats, goodputs)]
                model_num_unserved = model_num_unserved / np.sum(model_num_unserved)
                model_num_unserved_list[idx] = model_num_unserved

            group_configs = list(sol.group_configs)
            group_models = [list(x) for x in sol.group_models]

            # Merge two groups
            if np.random.uniform() < merge_group_prob:
                merge_two_groups(group_configs, group_models, model_num_unserved,
                                 weight_mem, mem_budget)

            # Split one group
            if np.random.uniform() < split_group_prob:
                split_one_group(group_configs, group_models, model_num_unserved,
                                weight_mem, mem_budget)

            # Mutate one model
            for g_id in range(len(group_models)):
                for m_id in range(len(group_models[g_id])):
                    if np.random.uniform() < mutate_one_model_prob:
                        new_m_id = np.random.choice(num_models, p=model_num_unserved)
                        if new_m_id not in group_models[g_id]:
                            group_models[g_id][m_id] = new_m_id

            new_sol = ModelPlacement(group_configs, group_models).normalize()
            next_sols.append(new_sol)
            visited.add(new_sol.group_models)

        if verbose >= 1:
            print(f"iter: {it}, best score: {best_score:.4f}, "
                  f"iter avg-score: {np.mean(scores):.4f}, "
                  f"iter #sol: {len(scores)}, "
                  f"visited #sol: {len(visited)}, "
                  f"elapsed: {time.time() - tic:.2f}, "
                  f"best sol: {best_sol}, ")

        it += 1
        cur_sols = next_sols + [best_sol]

    return best_sol


def merge_two_groups(group_configs, group_models, model_num_unserved,
                     weight_mem, mem_budget):
    retry = 0
    while retry < 10:
        g_id_1 = np.random.choice(len(group_models))
        g_id_2 = np.random.choice(len(group_models))
        if g_id_1 != g_id_2 and group_configs[g_id_1] == group_configs[g_id_2]:
            break
        retry += 1
    if retry >= 10:
        return

    # merge
    old_cfg = group_configs[g_id_1]
    new_cfg = ParallelConfig(old_cfg.dp, old_cfg.op, old_cfg.pp * 2)
    new_group_models = list(set(group_models[g_id_1] + group_models[g_id_2]))
    fit_mem_budget(new_cfg, new_group_models, model_num_unserved,
                   weight_mem, mem_budget)

    # update groups
    group_configs[g_id_1] = new_cfg
    group_models[g_id_1] = new_group_models
    del group_configs[g_id_2]
    del group_models[g_id_2]


def split_one_group(group_configs, group_models, model_num_unserved,
                    weight_mem, mem_budget):
    retry = 0
    while retry < 10:
        g_id = np.random.choice(len(group_models))
        if group_configs[g_id].pp % 2 == 0:
            break
        retry += 1
    if retry >= 10:
        return

    # split
    old_cfg = group_configs[g_id]
    new_cfg = ParallelConfig(old_cfg.dp, old_cfg.op, old_cfg.pp // 2)
    group_models[g_id].sort(key=lambda m_id: model_num_unserved[m_id])
    new_group_models_1 = group_models[g_id][::2]
    new_group_models_2 = group_models[g_id][1::2]

    fit_mem_budget(new_cfg, new_group_models_1, model_num_unserved,
                   weight_mem, mem_budget)
    fit_mem_budget(new_cfg, new_group_models_2, model_num_unserved,
                   weight_mem, mem_budget)

    group_configs[g_id] = new_cfg
    group_models[g_id] = new_group_models_1
    group_configs.append(new_cfg)
    group_models.append(new_group_models_2)


def fit_mem_budget(group_config, group_models, model_num_unserved,
                   weight_mem, mem_budget):
    # Remove models if necessary
    # Remove the model with the lowest number of unserved requests
    group_models.sort(key=lambda m_id: model_num_unserved[m_id])
    new_group_mem = sum(weight_mem[group_config][m_id] for m_id in group_models)
    while new_group_mem > mem_budget:
        m_id = group_models[0]
        del group_models[0]
        new_group_mem -= weight_mem[group_config][m_id]

    # Add models if possible
    # Add the model with the highest number of unserved requests
    model_ids = np.argsort(model_num_unserved)
    for m_id in reversed(model_ids):
        if m_id in group_models:
            continue
        if new_group_mem + weight_mem[group_config][m_id] <= mem_budget:
            group_models.append(m_id)
            new_group_mem += weight_mem[group_config][m_id]
            continue
        break
