"""Model placement policy"""
import dataclasses
import multiprocessing
import time
from typing import List, Tuple

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus

from alpa_serve.profiling import ParallelConfig, ProfilingResult


@dataclasses.dataclass
class ModelData:
    name: str
    slo: float
    average_load: float
    profiling_result: ProfilingResult


@dataclasses.dataclass
class ClusterEnv:
    num_devices: int
    mem_budget: float
    num_devices_per_node: int = 8


class PlacementPolicy:
    """The baseclass of placement policy"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.group_configs = None
        self.group_models = None
        self.debug_info = None

    def place_models(self, controller,
                     model_datas: List[ModelData], cluster_env: ClusterEnv):
        (self.group_configs, self.group_models, self.debug_info
         ) = self.solve_placement(model_datas, cluster_env)

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

        # Create model replicas
        for g_id in range(num_groups):
            for m_id in self.group_models[g_id]:
                name = model_datas[m_id].name
                controller.create_replica.remote(name, g_id, [self.group_configs[g_id]])

        if self.verbose:
            print(self.group_configs)
            print(self.group_models)
            print(self.debug_info)

        controller.sync()

    def __str__(self):
        group_strs = [f"({config}, {models})" for config, models
                      in zip(self.group_configs, self.group_models)]
        return f"{self.__class__.__name__}([" + ", ".join(group_strs) + "])"


class SelectiveReplication(PlacementPolicy):

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

        self.max_bs = 1
        self.time_limit = 20
        self.sum_k = 1e-4

    def compute_single_throuhput(self, model_data):
        parallel_config = ParallelConfig(1, 1, 1)
        stage_latency = model_data.profiling_result.para_dict[
            parallel_config].latency

        single_throughput = 0
        for b, (s,) in stage_latency.items():
            if b > self.max_bs:
                continue

            single_throughput = max(single_throughput, 1 / s)
        return single_throughput

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv):
        tic = time.time()

        # Load constants
        N = len(model_datas)
        M = cluster_env.num_devices
        C = cluster_env.mem_budget
        a = [x.average_load for x in model_datas]
        c = [x.profiling_result.para_dict[ParallelConfig(1,1,1)].weight_mem[0]
             for x in model_datas]
        t = [self.compute_single_throuhput(x) for x in model_datas]

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
        if self.verbose:
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

        return group_configs, group_models, {"objective": objective}


class ModelParallelismPlacement(PlacementPolicy):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

        self.time_limit = 20
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

    def compute_capability(self, model_data, parallel_config):
        slo = model_data.slo
        latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

        if latency_mem is None:
            return 0

        num_stages = parallel_config.pp
        max_cap = 0
        for b, ls in latency_mem.latency.items():
            if b > self.max_bs:
                continue

            # slo = sum(ls) + (n-1) * max(ls)
            # so, n = ceil((slo - sum(ls)) / max(ls)) + 1
            max_cap = max(max_cap, (slo - sum(ls)) // max(ls) + 1)

        return max_cap * (0.99 ** num_stages)

    def compute_max_stage_mem(self, model_data, parallel_config, mem_budget):
        latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

        if latency_mem is None:
            return mem_budget * 2

        return max(latency_mem.weight_mem)

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv):
        tic = time.time()

        # Load constants
        N = len(model_datas)
        M = cluster_env.num_devices
        C = cluster_env.mem_budget
        a = [x.average_load for x in model_datas]
        c = [x.profiling_result.para_dict[ParallelConfig(1,1,1)].weight_mem[0]
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
                f[i][k] = self.compute_capability(
                    model_data, parallel_config)
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
        if self.verbose:
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

        return group_configs, group_models, {"objective": objective}
