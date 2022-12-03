"""Selective replication."""
import logging
import multiprocessing
import time
from typing import List

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus
import ray

from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy.base_policy import (
    BasePlacementPolicy, ModelPlacement, ModelData, ClusterEnv,
    PlacementEvaluator, replica_placement_beam_search)
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.util import eps, inf


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

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = 1

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Load constants
        num_devices = cluster_env.num_devices
        num_models = len(model_datas)
        mem_budget = cluster_env.mem_budget

        parallel_config = ParallelConfig(1, 1, 1)
        weight_mem = [
            max(x.profiling_result.para_dict[parallel_config].weight_mem)
            for x in model_datas]
        single_throughput = [compute_single_throughput(x, self.max_bs) for x in model_datas]
        rate = [x.rate for x in model_datas]

        # Status variables
        burst_tolerance = np.zeros(num_models)
        used_mem = np.zeros(num_devices)
        model_set = [set() for _ in range(num_devices)]
        num_replicas = [0] * num_models

        # Optimization loop
        modified = True
        ct = 0
        while modified:
            modified = False
            model_ids = np.argsort(burst_tolerance)
            device_ids = np.argsort(used_mem)

            ## compute load
            #loads = [sum(rate[m_id]/num_replicas[m_id] for m_id in ms) for ms in model_set]
            #print(loads)

            # Greedily pick one model and a list of devices
            candidates = []
            for m_id in model_ids:
                for start_idx, d_id in enumerate(device_ids):
                    if (m_id not in model_set[d_id] and
                        weight_mem[m_id] + used_mem[d_id] <= mem_budget):
                        modified = True
                        if len(candidates):
                            if abs(used_mem[d_id] - used_mem[candidates[0]]) < eps:
                                candidates.append(d_id)
                        else:
                            candidates.append(d_id)

                if candidates:
                    break

            if modified:
                # Randomly pick one device
                ct += 1
                d_id = candidates[ct % len(candidates)]

                used_mem[d_id] += weight_mem[m_id]
                model_set[d_id].add(m_id)
                burst_tolerance[m_id] += single_throughput[m_id] / (rate[m_id] + eps)
                num_replicas[m_id] += 1

        # Parse solution
        group_configs = []
        group_models = []
        for i in range(num_devices):
            group_configs.append(parallel_config)
            group_models.append(list(model_set[i]))

        return (ModelPlacement(group_configs, group_models),
                {"objective": min(burst_tolerance)})


class SelectiveReplicationSearch(BasePlacementPolicy):

    def __init__(self,
                 simulation_duration: int = 1000,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = 1
        self.seed = 1234
        self.beam_size = 4
        self.simulation_duration = simulation_duration

        self.evaluator_method = "fast_simulator"
        self.parallel_evaluator = False

        if self.parallel_evaluator:
            ray.init(address="auto", ignore_reinit_error=True)

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            ws = []
            for i, data in enumerate(model_datas):
                ws.append(GammaProcess(data.rate, data.cv).generate_workload(
                    data.name, 0, duration=self.simulation_duration,
                    slo=data.slo, seed=self.seed + i))
            train_workload = Workload.merge(*ws)

        # Run beam search
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
            self.evaluator_method, self.parallel_evaluator)
        num_groups = cluster_env.num_devices
        sol = ModelPlacement([ParallelConfig(1,1,1)] * num_groups, [[] for _ in range(num_groups)])

        sol, debug_info = replica_placement_beam_search(
            sol, model_datas, cluster_env, train_workload,
            evaluator, self.beam_size, self.verbose)

        return sol, debug_info
