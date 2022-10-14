"""Model placement policy"""
from collections import namedtuple
from dataclasses import dataclass
import multiprocessing
import time
from typing import List, Tuple

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, lpDot, LpStatus


@dataclass
class ModelData:
    name: str
    model_mem: float
    average_load: float
    single_throughput: float
    pipeline_decay: List[Tuple[int, float]] = None


ParallelConfig = namedtuple("ParallelConfig", ("dp", "mp", "pp"))


class PlacementPolicy:
    pass


class SelectiveReplication(PlacementPolicy):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.time_limit = 20
        self.sum_k = 1e-4

    def place_models(self,
                     controller,
                     model_datas: List[ModelData],
                     num_gpus: int,
                     mem_budget: float):
        obj, placement = self.solve(model_datas, num_gpus, mem_budget)

        # Launch mesh groups
        for g_id in range(num_gpus):
            controller.launch_mesh_group_manager.remote(g_id, [1, 1])

        # Launch model replicas
        tasks = []
        for m_id in range(len(model_datas)):
            for g_id in range(num_gpus):
                if placement[m_id][g_id]:
                    name = model_datas[m_id].name
                    tasks.append(controller.create_replica.remote(
                        name, g_id, (ParallelConfig(1, 1, 1),)))
        return tasks

    def solve(self,
              model_datas: List[ModelData],
              num_gpus: int,
              mem_budget: float):
        tic = time.time()

        # Load constants
        N = len(model_datas)
        M = num_gpus
        C = 1
        a = [x.average_load for x in model_datas]
        c = [x.model_mem / mem_budget for x in model_datas]
        t = [x.single_throughput for x in model_datas]

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
            prob += lpSum(p[i][j] * c[i] for i in range(N)) <= C

        # (b). number of replicas
        for i in range(N):
            rep[i] = lpSum(p[i][j] for j in range(M))

        # (c). min tolerance and sum tolerance
        for i in range(N):
            prob += min_tolerance <= rep[i] * (t[i] / a[i])
        prob += sum_tolerance == lpSum(rep[i] for i in range(N))

        msg = self.verbose
        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")

        solver = pulp.PULP_CBC_CMD(mip=True,
                                   msg=msg,
                                   timeLimit=self.time_limit,
                                   threads=multiprocessing.cpu_count())
        prob.solve(solver)

        status = prob.status
        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        if self.verbose:
            print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
                  f"Time: {time.time() - tic}")

        if prob.status in [pulp.LpStatusInfeasible]:
            raise RuntimeError(
                "Cannot run the function under the given memory budget. "
                "Please increase the memory budget.")

        p_res = np.zeros((N, M), dtype=np.int8)
        for i in range(N):
            for j in range(M):
                if pulp.value(p[i][j]):
                    p_res[i][j] = 1

        return objective, p_res


class SelectiveReplicationWithPipeline(PlacementPolicy):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.time_limit = 20
        self.sum_k = 1e-4

    def place_models(self,
                     controller,
                     model_datas: List[ModelData],
                     num_gpus: int,
                     mem_budget: float,
                     group_sizes: List[int]):
        obj, (group_sizes, group_models) = self.solve(
            model_datas, num_gpus, mem_budget, group_sizes)
        num_groups = len(group_sizes)

        # Launch mesh groups
        for g_id in range(num_groups):
            pp_size = group_sizes[g_id]
            controller.launch_mesh_group_manager.remote(g_id, [1, pp_size])

        # Launch model replicas
        tasks = []
        for g_id in range(num_groups):
            pp_size = group_sizes[g_id]
            for m_id in group_models[g_id]:
                name = model_datas[m_id].name
                tasks.append(controller.create_replica.remote(
                    name, g_id, (ParallelConfig(1, 1, pp_size),)))

        return tasks

    def solve(self,
              model_datas: List[ModelData],
              num_gpus: int,
              mem_budget: float,
              group_sizes: List[int]):
        tic = time.time()

        # Load constants
        N = len(model_datas)
        M = num_gpus
        C = 1
        a = [x.average_load for x in model_datas]
        c = [x.model_mem / mem_budget for x in model_datas]
        t = [x.single_throughput for x in model_datas]

        G = num_gpus
        K = len(group_sizes)
        g = group_sizes
        d = np.zeros((N, K))
        for i in range(N):
            for group_size, decay in model_datas[i].pipeline_decay:
                try:
                    k = g.index(group_size)
                except ValueError:
                    k = -1

                if k >= 0:
                    d[i][k] = decay

        # 1. Create variables
        p = LpVariable.matrix("p", (range(N), range(G)), cat="Binary")
        rep = [None] * N
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
            prob += (lpSum(p[i][j] * c[i] for i in range(N)) <=
                     C * lpSum(s[j][k] * g[k] for k in range(K)))

        # (b). number of replicas
        for i in range(N):
            rep[i] = lpSum(pxs[i][j][k] * g[k] * d[i][k]
                           for j in range(G) for k in range(K))

        # (c). min tolerance and sum tolerance
        for i in range(N):
            prob += min_tolerance <= rep[i] * t[i] / a[i]

        prob += sum_tolerance == lpSum(rep[i] for i in range(N))

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

        msg = self.verbose
        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")

        solver = pulp.PULP_CBC_CMD(mip=True,
                                   msg=msg,
                                   timeLimit=self.time_limit,
                                   threads=multiprocessing.cpu_count())
        #solver = pulp.GLPK_CMD(mip=True, msg=msg)
        prob.solve(solver)

        status = prob.status
        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        if self.verbose:
            print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
                  f"Time: {time.time() - tic}")

        if prob.status in [pulp.LpStatusInfeasible]:
            raise RuntimeError(
                "Cannot run the function under the given memory budget. "
                "Please increase the memory budget.")

        s_res = []
        for j in range(G):
            assert sum(pulp.value(s[j][k]) for k in range(K)) == 1
            for k in range(K):
                if pulp.value(s[j][k]):
                    s_res.append(k)

        p_res = np.zeros((N, G), dtype=np.int8)
        for i in range(N):
            for j in range(G):
                if pulp.value(p[i][j]):
                    p_res[i][j] = 1

        group_sizes = []
        group_models = []
        for j in range(G):
            group_size = g[s_res[j]]
            if group_size:
                tmp = []
                for i in range(N):
                    if p_res[i][j]:
                        tmp.append(i)
                group_sizes.append(group_size)
                group_models.append(tmp)

        return objective, (group_sizes, group_models)
