from collections import namedtuple
from dataclasses import dataclass
import multiprocessing
import time
from typing import List

import numpy as np
import pulp
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, lpDot, LpStatus


@dataclass
class ModelData:
    name: str
    average_load: float
    model_mem: float
    single_throughput: float


ParallelConfig = namedtuple("ParallelConfig", ("dp", "mp", "pp"))


class PlacementPolicy:
    pass


class SelectiveReplication(PlacementPolicy):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def place_models(self,
                     controller,
                     mem_budget: float,
                     num_gpus: int,
                     model_infos: List[ModelData]):
        placement = self.solve(mem_budget, num_gpus, model_infos)

        for g_id in range(num_gpus):
            controller.launch_mesh_group_manager.remote(g_id, [1, 1])

        for g_id in range(num_gpus):
            for m_id in range(len(model_infos)):
                if placement[m_id][g_id]:
                    name = model_infos[m_id].name
                    controller.create_replica.remote(
                        name, g_id, (ParallelConfig(1, 1, 1),))

    def solve(self,
              mem_budget: float,
              num_gpus: int,
              model_infos: List[ModelData]):
        tic = time.time()

        num_models = len(model_infos)
        a = [x.average_load for x in model_infos]
        m = [x.model_mem for x in model_infos]
        s = [x.single_throughput for x in model_infos]

        # 1. Create variables
        p = LpVariable.matrix(
            "p", (range(num_models), range(num_gpus)), cat="Binary")
        rep = LpVariable.matrix(
            "rep", (range(num_models),), cat="Integer")
        min_tolerance = LpVariable("min_tolerance")

        # 2. Objective
        prob = LpProblem("myProblem", LpMaximize)
        obj = min_tolerance
        prob += obj

        # 3. Constraints
        # (a). memory budget on each GPU
        for j in range(num_gpus):
            prob += lpSum(p[i][j] * m[i] for i in range(num_models)) <= mem_budget

        # (b). number of replicas
        for i in range(num_models):
            prob += rep[i] == lpSum(p[i][j] for j in range(num_gpus))

        # (c). min tolerance
        for i in range(num_models):
            prob += min_tolerance <= rep[i] * s[i] / a[i]

        msg = self.verbose
        time_limit = 600
        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")

        solver = pulp.PULP_CBC_CMD(mip=True,
                                   msg=msg,
                                   timeLimit=time_limit,
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

        p_res = np.zeros((num_models, num_gpus), dtype=np.int8)
        for i in range(num_models):
            for j in range(num_gpus):
                if pulp.value(p[i][j]):
                    p_res[i][j] = 1

        return p_res
