"""Test placement policy"""
import unittest

import numpy as np

from alpa_serve.simulator.controller import Controller
from alpa_serve.placement_policy import (ModelData, ClusterEnv,
    SelectiveReplicationGreedy, SelectiveReplicationSearch,
    ModelParallelismGreedy, ModelParallelismSearch)
from alpa_serve.profiling import ParallelConfig, load_test_prof_result
from alpa.util import GB


class EchoModel:
    def __init__(self, parallel_config, virtual_mesh):
        pass

    async def handle_request(self, request):
        return request


class PlacementPolicyTest(unittest.TestCase):

    def test_selective_replication(self):
        cluster_env = ClusterEnv(num_devices=4, mem_budget=4.5*GB)
        model_datas = [
            ModelData("m0", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m1", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
        ]

        for policy in [SelectiveReplicationGreedy(),
                       SelectiveReplicationSearch(verbose=1)]:
            placement, _ = policy.solve_placement(
                model_datas, cluster_env)

            # Check result
            assert all(g == ParallelConfig(1, 1, 1) for g in placement.group_configs)
            for i in range(4):
                assert sum(x.count(i) for x in placement.group_models) == 2

    def test_model_parallelism(self):
        cluster_env = ClusterEnv(num_devices=4, mem_budget=4.5*GB)
        model_datas = [
            ModelData("m0", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m1", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
        ]

        for policy in [ModelParallelismGreedy(group_size=2)]:
            placement, _ = policy.solve_placement(
                model_datas, cluster_env)

            assert len(placement.group_configs) == 2
            assert placement.group_configs[0].pp == 2
            assert placement.group_configs[1].pp == 2
            assert placement.group_models[0] == [0, 1, 2, 3]
            assert placement.group_models[1] == [0, 1, 2, 3]

    def test_model_parallelism_search(self):
        cluster_env = ClusterEnv(num_devices=4, mem_budget=2.5*GB)
        model_datas = [
            ModelData("m0", 0.4, 4, 8, load_test_prof_result("test-2GB-100ms")),
            ModelData("m1", 0.4, 4, 8, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 0.4, 4, 8, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 0.4, 4, 8, load_test_prof_result("test-2GB-100ms")),
        ]

        for policy in [ModelParallelismSearch(verbose=2)]:
            placement, _ = policy.solve_placement(
                model_datas, cluster_env)

            assert len(placement.group_configs) == 1
            assert placement.group_configs[0].pp == 4
            assert list(placement.group_models[0]) == [0, 1, 2, 3]

    def test_placement_api(self):
        for policy in [SelectiveReplicationGreedy(), ModelParallelismGreedy()]:
            controller = Controller()
            controller.register_model.remote("m0", EchoModel)
            controller.register_model.remote("m1", EchoModel)
            controller.register_model.remote("m2", EchoModel)
            controller.register_model.remote("m3", EchoModel)

            cluster_env = ClusterEnv(num_devices=4, mem_budget=4.5*GB)
            model_datas = [
                ModelData("m0", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
                ModelData("m1", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
                ModelData("m2", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
                ModelData("m3", 1, 5, 1, load_test_prof_result("test-2GB-100ms")),
            ]
            policy.place_models(controller, cluster_env, model_datas)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PlacementPolicyTest("test_selective_replication"))
    suite.addTest(PlacementPolicyTest("test_model_parallelism"))
    suite.addTest(PlacementPolicyTest("test_model_parallelism_search"))
    suite.addTest(PlacementPolicyTest("test_placement_api"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
