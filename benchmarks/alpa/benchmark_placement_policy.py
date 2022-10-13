from collections import namedtuple

import numpy as np

from alpa_serve.placement_policy import (SelectiveReplication, ModelData,
    SelectiveReplicationWithPipeline)
from alpa.util import GB, to_str_round


CaseInput = namedtuple("CaseInput", [
    "num_models", "num_gpus", "model_size", "load_distribution"])


def generate_cases():
    cases = []

    num_gpus = 8
    model_size = 3
    num_models = [1, 2, 4, 8, 16, 32]

    for n in num_models:
        cases.append(CaseInput(n, num_gpus, model_size, "uniform"))

    for n in num_models:
        cases.append(CaseInput(n, num_gpus, model_size, "power"))

    return cases


def bench_cases(cases):
    pipeline_decay = [(1, 1), (2, 0.95), (4, 0.9)]
    mem_budgets = [2, 4, 8, 16, 32, 64]
    unit = 1.0

    sr = SelectiveReplication()
    srp = SelectiveReplicationWithPipeline()

    res = np.zeros((len(cases), len(mem_budgets)))

    for i, case in enumerate(cases):
        print(f"\n----- {case} -----")
        num_models, num_gpus, model_size, load_distribution = case

        if load_distribution == "uniform":
            average_load = [unit] * num_models
        elif load_distribution == "power":
            total_load = unit * num_models
            average_load = [total_load * (0.5 ** i) for i in range(num_models)]
        else:
            raise ValueError(f"Invalid distribution: {load_distribution}")

        model_datas = [
            ModelData(f"m{i}", model_size, average_load[i], unit, pipeline_decay)
            for i in range(num_models)
        ]

        for j, mem_budget in enumerate(mem_budgets):
            obj1, _ = sr.solve(model_datas, num_gpus, mem_budget)
            obj2, (group_sizes, group_models) = srp.solve(
                model_datas, num_gpus, mem_budget, [0, 1, 2, 4])
            #obj2 = obj1

            if obj1 <= 1e-5:
                gain = float("inf")
            else:
                gain = obj2 / obj1

            print(f"mem_budget = {mem_budget}, obj1 = {obj1}, obj2 = {obj2}")
            print(f"group sizes = {group_sizes}, group models = {group_models}\n")

            res[i][j] = gain

    for row in res:
        print(to_str_round(row, 2))


if __name__ == "__main__":
    cases = generate_cases()

    bench_cases(cases)
