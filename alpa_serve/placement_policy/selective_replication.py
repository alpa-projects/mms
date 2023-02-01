"""Selective replication."""
import logging
import multiprocessing
import time
from typing import List

import numpy as np
import ray

from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy.base_policy import (
    BasePlacementPolicy, ModelPlacement, ModelData, ClusterEnv,
    PlacementEvaluator, gen_train_workload, ModelPlacementWithReplacement,
    replica_placement_fast_greedy, replica_placement_beam_search)
from alpa_serve.simulator.workload import Workload
from alpa_serve.util import eps, inf, to_str_round


def compute_single_throughput(model_data, max_bs):
    parallel_config = ParallelConfig(1, 1, 1)
    stage_latency = model_data.profiling_result.para_dict[
        parallel_config].latency

    single_throughput = 0
    for b, (s,) in stage_latency.items():
        if b > max_bs:
            continue

        single_throughput = max(single_throughput, 1 / s)
    return single_throughput


class SelectiveReplicationILP(BasePlacementPolicy):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = 1
        self.time_limit = 30
        self.sum_k = 1e-4

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        import pulp
        from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus

        tic = time.time()

        # Load constants
        N = len(model_datas)
        M = cluster_env.num_devices
        C = cluster_env.mem_budget
        a = [x.rate for x in model_datas]
        c = [x.profiling_result.para_dict[ParallelConfig(1, 1, 1)].weight_mem[0]
             for x in model_datas]
        t = [compute_single_throughput(x, self.max_bs) for x in model_datas]

        # 1. Create variables
        p = LpVariable.matrix("p", (range(N), range(M)), cat="Binary")
        rep = [None] * N
        min_tolerance = LpVariable("min_tolerance", lowBound=0)
        sum_tolerance = LpVariable("sum_tolerance", lowBound=0)

        # 2. Objective
        prob = LpProblem("myProblem", LpMaximize)
        obj = min_tolerance + self.sum_k * sum_tolerance
        prob += obj

        # 3. Constraints
        # (a). memory budget on each GPU
        for j in range(M):
            prob += lpSum(p[i][j] * (c[i] / C) for i in range(N)) <= 1

        # (b). number of replicas
        for i in range(N):
            rep[i] = lpSum(p[i][j] for j in range(M))

        # (c). min tolerance and sum tolerance
        for i in range(N):
            prob += min_tolerance <= rep[i] * (t[i] / a[i])
        prob += sum_tolerance == lpSum(rep[i] * (t[i] / a[i]) for i in range(N))

        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")

        solver = pulp.PULP_CBC_CMD(mip=True,
                                   msg=False,
                                   timeLimit=self.time_limit,
                                   threads=multiprocessing.cpu_count())
        prob.solve(solver)

        status = prob.status
        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        if self.verbose >= 2:
            print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
                  f"Time: {time.time() - tic}")

        if prob.status in [pulp.LpStatusInfeasible]:
            raise RuntimeError(
                "Cannot run the function under the given memory budget. "
                "Please increase the memory budget.")

        # Parse solution
        p_res = np.zeros((N, M), dtype=np.int8)
        for i in range(N):
            for j in range(M):
                if pulp.value(p[i][j]):
                    p_res[i][j] = 1

        group_configs = []
        group_models = []
        for j in range(M):
            tmp = []
            for i in range(N):
                if p_res[i][j]:
                    tmp.append(i)
            group_configs.append(ParallelConfig(1, 1, 1))
            group_models.append(tmp)

        return ModelPlacement(group_configs, group_models), {"objective": objective}


class SelectiveReplicationGreedy(BasePlacementPolicy):

    def __init__(self, use_evo_search: bool = False, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.use_evo_search = use_evo_search

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        # Run greedy placement
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                                       "fast_simulator", False)
        num_groups = cluster_env.num_devices
        sol = ModelPlacement([ParallelConfig(1,1,1)] * num_groups, [[] for _ in range(num_groups)])

        sol = replica_placement_fast_greedy(
            sol, model_datas, cluster_env, train_workload,
            evaluator, self.verbose)

        if self.use_evo_search:
            sol = evolutionary_search([sol], model_datas, evaluator, self.verbose)
        return sol, None


class SelectiveReplicationUniform(BasePlacementPolicy):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        num_models = len(model_datas)
        model_memory = model_datas[0].profiling_result.para_dict[ParallelConfig(1, 1, 1)].weight_mem[0]
        num_models_per_group = min(int(cluster_env.mem_budget / model_memory), num_models)
        num_groups = cluster_env.num_devices

        group_models = []
        for i in range(num_groups):
            group = np.arange(i, i + num_models_per_group) % num_models
            group_models.append(group)
        sol = ModelPlacement([ParallelConfig(1,1,1)] * num_groups,
                             group_models)
        return sol, None


class SelectiveReplicationSearch(BasePlacementPolicy):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.beam_size = 3

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        # Run beam search
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                                       "fast_simulator", False)
        num_groups = cluster_env.num_devices
        sol = ModelPlacement([ParallelConfig(1,1,1)] * num_groups, [[] for _ in range(num_groups)])

        sol = replica_placement_beam_search(
            sol, model_datas, cluster_env, train_workload,
            evaluator, self.beam_size, self.verbose)
        return sol, None


class SelectiveReplicationReplacement(BasePlacementPolicy):

    def __init__(self, replacement_interval: int,
                 use_evo_search: bool = False, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.replacement_interval = replacement_interval
        self.use_evo_search = use_evo_search

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        ws = train_workload.split_time_interval(self.replacement_interval)

        start_times = []
        placements = []
        for i in range(len(ws)):
            # Run greedy placement
            evaluator = PlacementEvaluator(model_datas, cluster_env, ws[i],
                                           "fast_simulator", False)
            num_groups = cluster_env.num_devices
            sol = ModelPlacement([ParallelConfig(1,1,1)] * num_groups, [[] for _ in range(num_groups)])

            sol = replica_placement_fast_greedy(
                sol, model_datas, cluster_env, ws[i],
                evaluator, self.verbose)

            if self.use_evo_search:
                sol = evolutionary_search([sol], model_datas, evaluator, self.verbose)

            start_times.append(ws[i].arrivals[0])
            placements.append(sol)

        return ModelPlacementWithReplacement(start_times, placements), None

