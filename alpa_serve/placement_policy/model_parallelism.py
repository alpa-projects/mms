"""Selective replication with model parallelism."""
from collections import namedtuple
from functools import partial
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
    BasePlacementPolicy, ModelData, ClusterEnv, ModelPlacement,
    PlacementEvaluator, replica_placement_beam_search)
from alpa_serve.simulator.controller import simulate_one_case
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.trace import Trace
from alpa_serve.util import get_factors, ServingCase, eps


def compute_capability(model_data, parallel_config, max_bs):
    slo = model_data.slo
    latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

    if latency_mem is None:
        return 0

    num_stages = parallel_config.pp
    max_cap = 0
    for b, ls in latency_mem.latency.items():
        if b > max_bs:
            continue

        # slo = sum(ls) + (n-1) * max(ls)
        # so, n = ceil((slo - sum(ls)) / max(ls)) + 1
        max_cap = max(max_cap, (slo - sum(ls)) // max(ls) + 1)

    return max_cap * (0.99 ** num_stages)


class ModelParallelismILP(BasePlacementPolicy):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.time_limit = 30
        self.sum_k = 1e-4
        self.max_bs = 1

        # Hard coded for now. Expose this as parameters later
        self.group_configs = [
            ParallelConfig(0, 0, 0),
            ParallelConfig(1, 1, 1),
            ParallelConfig(1, 1, 2),
            ParallelConfig(1, 1, 4),
            ParallelConfig(1, 1, 8),
        ]
        self.group_sizes = [
            np.prod(x) for x in self.group_configs
        ]

    def compute_max_stage_mem(self, model_data, parallel_config, mem_budget):
        latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

        if latency_mem is None:
            return mem_budget * 2

        return max(latency_mem.weight_mem)

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

        G = cluster_env.num_devices
        K = len(self.group_configs)
        g = self.group_sizes
        f = np.zeros((N, K))
        d = np.zeros((N, K))
        for i in range(N):
            model_data = model_datas[i]
            for k in range(K):
                parallel_config = self.group_configs[k]
                f[i][k] = compute_capability(model_data, parallel_config, self.max_bs)
                d[i][k] = self.compute_max_stage_mem(
                    model_data, parallel_config, cluster_env.mem_budget)

        # 1. Create variables
        p = LpVariable.matrix("p", (range(N), range(G)), cat="Binary")
        cap = [None] * N
        min_tolerance = LpVariable("min_tolerance", lowBound=0)
        sum_tolerance = LpVariable("sum_tolerance", lowBound=0)
        s = LpVariable.matrix("s", (range(G), range(K)), cat="Binary")
        pxs = LpVariable.matrix("pxs", (range(N), range(G), range(K)), cat="Binary")

        # 2. Objective
        prob = LpProblem("myProblem", LpMaximize)
        obj = min_tolerance + self.sum_k * sum_tolerance
        prob += obj

        # 3. Constraints
        # (a). memory budget on each GPU
        for j in range(G):
            prob += (lpSum(p[i][j] * (c[i] / C) for i in range(N)) <=
                     lpSum(s[j][k] * g[k] for k in range(K)))

        ## A more precise version, not used right now
        #for j in range(G):
        #    prob += (lpSum(pxs[i][j][k] * (d[i][k] / C)
        #                   for i in range(N) for k in range(K)) <= 1)

        # (b). capability
        for i in range(N):
            cap[i] = lpSum(pxs[i][j][k] * f[i][k]
                           for j in range(G) for k in range(K))

        # (c). min tolerance and sum tolerance
        for i in range(N):
            prob += min_tolerance <= cap[i] / a[i]

        prob += sum_tolerance == lpSum(cap[i] / a[i] for i in range(N))

        # (d). group size
        prob += lpSum(s[j][k] * g[k] for j in range(G) for k in range(K)) == M

        # (e). only one configuration
        for j in range(G):
            prob += lpSum(s[j][k] for k in range(K)) == 1

        # (f). linearization
        for i in range(N):
            for j in range(G):
                for k in range(K):
                    prob += pxs[i][j][k] <= p[i][j]
                    prob += pxs[i][j][k] <= s[j][k]
                    prob += pxs[i][j][k] >= p[i][j] + s[j][k] - 1

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

        # Group configuration selection
        s_res = []
        for j in range(G):
            assert sum(pulp.value(s[j][k]) for k in range(K)) == 1
            for k in range(K):
                if pulp.value(s[j][k]):
                    s_res.append(k)

        # Placement
        p_res = np.zeros((N, G), dtype=np.int8)
        for i in range(N):
            for j in range(G):
                if pulp.value(p[i][j]):
                    p_res[i][j] = 1

        group_configs = []
        group_models = []
        for j in range(G):
            config_id = s_res[j]
            if self.group_sizes[config_id]:
                tmp = []
                for i in range(N):
                    if p_res[i][j]:
                        tmp.append(i)
                group_configs.append(self.group_configs[config_id])
                group_models.append(tmp)

        return ModelPlacement(group_configs, group_models), {"objective": objective}


class ModelParallelismGreedy(BasePlacementPolicy):

    def __init__(self, group_size: int = 2, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = 1
        self.group_size = group_size

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Load constants
        num_devices = cluster_env.num_devices
        num_models = len(model_datas)
        mem_budget = cluster_env.mem_budget

        if num_devices % self.group_size != 0:
            return ModelPlacement([], []), None

        assert num_devices % self.group_size == 0
        num_groups = num_devices // self.group_size

        parallel_config = ParallelConfig(1, 1, self.group_size)
        weight_mem = [
            max(x.profiling_result.para_dict[parallel_config].weight_mem)
            for x in model_datas]
        single_throughput = [
            compute_capability(x, parallel_config, self.max_bs)
            for x in model_datas]
        rate = [x.rate for x in model_datas]

        if max(single_throughput) <= 1e-5:
            single_throughput = [1e-5] * len(single_throughput)

        # Status variables
        burst_tolerance = np.zeros(num_models)
        used_mem = np.zeros(num_groups)
        model_set = [set() for _ in range(num_groups)]

        # Optimization loop
        modified = True
        ct = 0
        while modified:
            modified = False
            model_ids = np.argsort(burst_tolerance)
            group_ids = np.argsort(used_mem)

            # Greedly pick one model and a list of groups
            candidates = []
            for m_id in model_ids:
                for start_idx, g_id in enumerate(group_ids):
                    if (m_id not in model_set[g_id] and
                        weight_mem[m_id] + used_mem[g_id] <= mem_budget):
                        modified = True
                        if len(candidates):
                            if abs(used_mem[g_id] - used_mem[candidates[0]]) < eps:
                                candidates.append(g_id)
                        else:
                            candidates.append(g_id)

                if candidates:
                    break

            if modified:
                # Randomly pick one group
                ct += 1
                g_id = candidates[ct % len(candidates)]

                used_mem[g_id] += weight_mem[m_id]
                model_set[g_id].add(m_id)
                burst_tolerance[m_id] += single_throughput[m_id] / (rate[m_id] + eps)

        # Parse solution
        group_configs = []
        group_models = []
        for i in range(num_groups):
            group_configs.append(parallel_config)
            group_models.append(list(model_set[i]))

        return (ModelPlacement(group_configs, group_models),
                {"objective": min(burst_tolerance)})


class ModelParallelismSearch(BasePlacementPolicy):

    def __init__(self,
                 max_bs: int = 1,
                 max_pp: int = 8,
                 max_op: int = 4,
                 n_iter: int = 1,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = max_bs
        self.max_pp = max_pp
        self.max_op = max_op
        self.n_iter = n_iter
        self.seed = 0
        self.beam_size = 4
        self.simulation_min_duration = 100
        self.simulation_min_samples = 30000

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
            total_rate = sum(d.rate for d in model_datas)
            duration = max(self.simulation_min_duration,
                           self.simulation_min_samples / total_rate)

            ws = []
            for i, data in enumerate(model_datas):
                ws.append(GammaProcess(data.rate, data.cv).generate_workload(
                    data.name, 0, duration=duration,
                    slo=data.slo, seed=self.seed + i))
            train_workload = Workload.merge(*ws)

        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
            self.evaluator_method, self.parallel_evaluator)

        # Get initial solutions
        initial_sols = self.enumerate_group_configs(cluster_env)
        for i in range(len(initial_sols)):
            initial_sols[i] = replica_placement_beam_search(
                initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                self.beam_size, self.verbose)[0]
            #initial_sols[i] = self.greedy_placement(
            #     initial_sols[i], model_datas, cluster_env, train_workload)

        # Iterative search
        cur_sols = initial_sols
        best_score = -1
        best_sol = None

        it = 0
        tic = time.time()
        while it < self.n_iter:
            scores = evaluator.get_scores(cur_sols)

            tmp_best_idx = np.argmax(scores)

            if scores[tmp_best_idx] > best_score:
                best_score = scores[tmp_best_idx]
                best_sol = cur_sols[tmp_best_idx]

            if self.verbose >= 1:
                print(f"iter: {it}, best score: {best_score}, "
                      f"iter score: {scores[tmp_best_idx]}, "
                      f"iter #sol: {len(scores)}, "
                      f"elapsed: {time.time() - tic:.2f}, "
                      f"best placement: {best_sol}, ")

            if self.verbose >= 2:
                print("\n--- iter sols ---")
                for i in range(len(cur_sols)):
                    print(f"idx={i}")
                    print(f"placement={cur_sols[i]}")
                    print(f"score={scores[i]:.3f}\n")
                print("-----------------")

            # TODO: mutate solution
            it += 1

        return best_sol, {}

    def enumerate_group_configs(self, cluster_env):
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get_factors(num_devices):
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue

            for pp in get_factors(group_size):
                op = group_size // pp
                num_groups = num_devices // group_size

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_groups,
                                           [[] for _ in range(num_groups)]))
        return sols

    def greedy_placement(self, sol: ModelPlacement,
                         model_datas: List[ModelData],
                         cluster_env: ClusterEnv,
                         train_workload: Workload):
        num_models = len(model_datas)
        mem_budget = cluster_env.mem_budget
        group_configs = sol.group_configs
        inf = 1e20

        num_groups = len(group_configs)

        weight_mem = {}  # Dict[parallel_config -> [model_idx -> weight_mem]]
        for parallel_config in sol.group_configs:
            weight_mem[parallel_config] = [
                max(x.profiling_result.para_dict[parallel_config].weight_mem)
                if parallel_config in x.profiling_result.para_dict
                else inf
                for x in model_datas]

        single_throughput = {}  # Dict[parallel_config -> [model_idx -> capability]]
        for parallel_config in sol.group_configs:
            tmp_throughput = [
                compute_capability(x, parallel_config, self.max_bs)
                for x in model_datas]
            if max(tmp_throughput) <= 1e-5:
                single_throughput[parallel_config] = [1e-5] * len(tmp_throughput)
            else:
                single_throughput[parallel_config] = tmp_throughput

        rate = [x.rate for x in model_datas]

        # Status variables
        burst_tolerance = np.zeros(num_models)
        used_mem = np.zeros(num_groups)
        model_set = [set() for _ in range(num_groups)]

        # Optimization loop
        modified = True
        ct = 0
        while modified:
            modified = False
            model_ids = np.argsort(burst_tolerance)
            group_ids = np.argsort(used_mem)

            # Greedly pick one model and a list of groups
            candidates = []
            for m_id in model_ids:
                for start_idx, g_id in enumerate(group_ids):
                    parallel_config = group_configs[g_id]
                    if (m_id not in model_set[g_id] and
                        weight_mem[parallel_config][m_id] + used_mem[g_id] <= mem_budget):
                        modified = True
                        if len(candidates):
                            if abs(used_mem[g_id] - used_mem[candidates[0]]) < eps:
                                candidates.append(g_id)
                        else:
                            candidates.append(g_id)

                if candidates:
                    break

            if modified:
                # Randomly pick one group
                ct += 1
                g_id = candidates[ct % len(candidates)]
                parallel_config = group_configs[g_id]

                used_mem[g_id] += weight_mem[parallel_config][m_id]
                model_set[g_id].add(m_id)
                burst_tolerance[m_id] += (
                    single_throughput[parallel_config][m_id] / (rate[m_id] + eps))

        # Parse solution
        group_models = []
        for i in range(num_groups):
            group_models.append(list(model_set[i]))

        return ModelPlacement(group_configs, group_models)
