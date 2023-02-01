"""Selective replication with model parallelism."""
from collections import namedtuple
from functools import partial
import logging
import math
import multiprocessing
import time
from typing import List, Tuple

import numpy as np
import ray

from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy.base_policy import (
    BasePlacementPolicy, ModelData, ClusterEnv, ModelPlacement,
    PlacementEvaluator, gen_train_workload,
    replica_placement_round_robin,
    replica_placement_fast_greedy, replica_placement_beam_search,
    replica_placement_on_last_group, evolutionary_search)
from alpa_serve.simulator.controller import simulate_one_case
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.trace import Trace
from alpa_serve.util import (
    get_factors, get_partitions, get2tok, decompose2tok,
    ServingCase, eps)


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

    def __init__(self, group_size: int = 2,
                 use_evo_search: bool = False,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.group_size = group_size
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

        assert cluster_env.num_devices % self.group_size == 0
        num_groups = cluster_env.num_devices // self.group_size
        sol = ModelPlacement([ParallelConfig(1,1,self.group_size)] * num_groups,
                             [[] for _ in range(num_groups)])
        sol = replica_placement_fast_greedy(
            sol, model_datas, cluster_env, train_workload,
            evaluator, self.verbose)

        if self.use_evo_search:
            sol = evolutionary_search([sol], model_datas, cluster_env,
                                      evaluator, 200, self.verbose)
        return sol, None


def solve_separation_placement(self,
                               eco_separation: List[Tuple[List[ModelData], ClusterEnv]],
                               model_id_map,
                               train_workload: Workload):
    sol = ModelPlacement([],[])
    for i, eco in enumerate(eco_separation):
        sub_model_datas, sub_cluster_env = eco
        eco_sol, _ = self.solve_placement_one_eco(sub_model_datas, sub_cluster_env, train_workload)
        sol.group_configs += eco_sol.group_configs
        sol.group_models += [[model_id_map[(i, model_id)] for model_id in group]
                             for group in eco_sol.group_models]
    return sol


class ModelParallelismRR(BasePlacementPolicy):

    def __init__(self,
                 verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_bs = max_bs
        self.max_pp = max_pp
        self.max_op = max_op

        self.evaluator_method = "fast_simulator"


    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        # parallel config (dp = 1, op = 1, pp = 4)
        num_reg_groups = cluster_env.num_devices // 4
        quo_groups = decompose2tok(cluster_env.num_devices % 4)
        init_sol = ModelPlacement([ParallelConfig(1, 1, 4)] * num_reg_groups +
                                  [ParallelConfig(1, 1, s) for s in quo_groups],
                                  [[] for _ in range(num_reg_groups + len(quo_groups))])

        sol = replica_placement_round_robin(
                   init_sol, model_datas, cluster_env, train_workload, self.verbose)

        return sol, {}


class ModelParallelismSearch(BasePlacementPolicy):

    def __init__(self,
                 max_bs: int = 1,
                 max_pp: int = 8,
                 max_op: int = 4,
                 use_evo_search: bool = False,
                 use_separation: bool = False,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = max_bs
        self.max_pp = max_pp
        self.max_op = max_op
        self.n_iter = 1
        self.seed = 0
        self.beam_size = 3
        self.use_evo_search = use_evo_search
        self.use_separation = use_separation

        self.evaluator_method = "fast_simulator"
        self.parallel_evaluator = False
        self.parallel_initial_placement = False

        if ((self.parallel_evaluator or self.parallel_initial_placement)
            and not ray.is_initialized()):
            ray.init(address="auto", ignore_reinit_error=True)


    def solve_placement_one_eco(self,
                                model_datas: List[ModelData],
                                cluster_env: ClusterEnv,
                                train_workload: Workload = None):
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
            self.evaluator_method, self.parallel_evaluator)

        # Get initial solutions
        initial_sols = self.enumerate_group_configs_uneven(cluster_env)

        if self.parallel_initial_placement:
            func = ray.remote(replica_placement_fast_greedy).remote
            for i in range(len(initial_sols)):
                initial_sols[i] = func(
                    initial_sols[i], model_datas, cluster_env, train_workload, None,
                    self.verbose)
            initial_sols = ray.get(initial_sols)
        else:
            for i in range(len(initial_sols)):
                initial_sols[i] = replica_placement_fast_greedy(
                    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                    self.verbose)
                #initial_sols[i] = replica_placement_beam_search(
                #    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                #     self.beam_size, self.verbose)

        scores = evaluator.get_scores(initial_sols)
        best_idx = np.argmax(scores)
        best_sol = initial_sols[best_idx]

        return best_sol, {}


    def enumerate_separations(self,
                              model_datas: List[ModelData],
                              cluster_env: ClusterEnv):
        same_model_threshold = 0.38

        model_id_map = {}
        eco_model_datas = []
        cluster_latencies = []
        for model_id, model_data in enumerate(model_datas):
            cur_latency = max(model_data.profiling_result. \
                          para_dict[ParallelConfig(1, 1, 1)].latency[1])
            flag = False
            for i, cluster in enumerate(eco_model_datas):
                cluster_latency = max(cluster[0].profiling_result. \
                                  para_dict[ParallelConfig(1, 1, 1)].latency[1])
                if math.fabs(cur_latency - cluster_latency) / cluster_latency < same_model_threshold:
                    model_id_map[(i, len(cluster))] = model_id
                    cluster.append(model_data)
                    flag = True
                    break
            if not flag:
                model_id_map[(len(eco_model_datas), 0)] = model_id
                eco_model_datas.append([model_data])
                cluster_latencies.append(cur_latency)

        # List[List[(List[ModelData], ClusterEnv)]]
        partitions = get_partitions(cluster_env.num_devices, len(eco_model_datas))

        ## reduce num partitions
        ratio = np.empty(len(eco_model_datas), dtype=np.float32)
        for i, eco_model_data in enumerate(eco_model_datas):
            ratio[i] = sum(x.rate for x in eco_model_data)
        ratio = ratio / np.sum(ratio)   # q/s

        for threshold in [1.0, 0.5, 0.3, 0.2, 0.1]:
            reduced_partitions = []
            for partition in partitions:
                throughputs = [x / l for x, l in zip(partition, cluster_latencies)]   # q/s
                norm_throughputs = np.array(throughputs) / sum(throughputs)
                dis = np.max(np.abs(ratio - norm_throughputs))
                if dis < threshold:
                    reduced_partitions.append(partition)

            if len(reduced_partitions) < 100:
                break

        print(f"original: {len(partitions)}  reduced: {len(reduced_partitions)}")

        separations = [[(eco_model_datas[i], ClusterEnv(device_cnt, cluster_env.mem_budget)) \
                        for i, device_cnt in enumerate(partition)] \
                       for partition in reduced_partitions]

        return separations, model_id_map


    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        best_sol, _ = self.solve_placement_one_eco(model_datas, cluster_env, train_workload)

        # Separate unequal model
        if self.use_separation:
            eco_separations, model_id_map = self.enumerate_separations(model_datas, cluster_env)
            print("number of combinations: ", len(eco_separations))

            parallel = False
            if parallel:
                func = ray.remote(solve_separation_placement).remote
            else:
                func = solve_separation_placement

            sols = []
            for eco_separation in eco_separations:
                sols.append(func(self, eco_separation, model_id_map, train_workload))

            if parallel:
                sols = ray.get(sols)

            evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                self.evaluator_method, self.parallel_evaluator)
            scores = evaluator.get_scores(sols)
            best_idx = np.argmax(scores)

            evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                self.evaluator_method, self.parallel_evaluator)
            score_mixed = evaluator.get_scores([best_sol])[0]

            print(f"score_mixed: {score_mixed:.3f}, score_separate: {scores[best_idx]:.3f}")
            if scores[best_idx] > score_mixed:
                best_sol = sols[best_idx]

        if self.use_evo_search:
            best_sol = evolutionary_search(
                [best_sol], model_datas, cluster_env,
                evaluator, 200, self.verbose)
        return best_sol, {}


    def enumerate_group_configs_uneven(self, cluster_env: ClusterEnv):
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get2tok(num_devices):
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue
            num_reg_groups = num_devices // group_size
            quo_groups = decompose2tok(num_devices % group_size)

            for pp in get_factors(group_size):
                op = group_size // pp

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_reg_groups +
                                           [ParallelConfig(1, 1, s) for s in quo_groups],
                                           [[] for _ in range(num_reg_groups + len(quo_groups))]))
        return sols


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

    def greedy_group_configs(self,
                             model_datas: List[ModelData],
                             cluster_env: ClusterEnv,
                             train_workload: Workload,
                             evaluator: PlacementEvaluator,
                             beam_size = 3):

        assert beam_size >= 1, "beam size should >= 1."

        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        beam_sols = [[ModelPlacement([], [])]]

        for cur_num in range(1, num_devices + 1):
            ## solve sols[cur_num]
            next_sols = []
            for last_group_size in range(1, (cur_num - 1) % num_devices_per_node + 1 + 1):
                ## solve from sols[cur_num - last_group_size]
                # print("last_group_size ", last_group_size)
                for pp in get_factors(last_group_size):
                    op = last_group_size // pp
                    if pp > self.max_pp or op > self.max_op:
                        continue

                    for sol in beam_sols[cur_num - last_group_size]:
                        pre_sol = sol.copy()
                        pre_sol.group_configs.append(ParallelConfig(1, op, pp))
                        pre_sol.group_models = [[] for _ in range(len(pre_sol.group_configs))]

                        #new_sol = replica_placement_on_last_group(
                        #new_sol = replica_placement_beam_search(
                        #              pre_sol, model_datas, cluster_env, train_workload,
                        #              evaluator, self.beam_size, self.verbose)
                        new_sol = replica_placement_fast_greedy(
                                      pre_sol, model_datas, cluster_env, train_workload,
                                      evaluator, self.verbose)
 
                        next_sols.append(new_sol)
            scores = evaluator.get_scores(next_sols)
            next_indices = np.argsort(scores)[::-1][:beam_size]
            beam_sols.append([])
            for i in range(len(next_indices)):
                beam_sols[cur_num].append(next_sols[next_indices[i]])

        return beam_sols[num_devices]

class ModelParallelismEqual(BasePlacementPolicy):
    
    def __init__(self, pp, op, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.pp = pp
        self.op = op

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        group_size = self.op * self.pp
        num_groups = cluster_env.num_devices // group_size
        sol = ModelPlacement([ParallelConfig(1, self.op, self.pp)] * num_groups, [[i] for i in range(num_groups)])

        return sol, None
