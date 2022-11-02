"""Test placement policy"""
import unittest

import numpy as np

from alpa_serve.simulator.controller import Controller
from alpa_serve.placement_policy import (ModelData, ClusterEnv,
    SelectiveReplication, ModelParallelismPlacement)
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
            ModelData("m1", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m4", 1, 5, load_test_prof_result("test-2GB-100ms")),
        ]

        policy = SelectiveReplication()
        group_configs, group_models, _ = policy.solve_placement(
            model_datas, cluster_env)

        # Check result
        assert all(g == ParallelConfig(1, 1, 1) for g in group_configs)
        for i in range(4):
            assert sum(x.count(i) for x in group_models) == 2

    def test_selective_replication_with_pipeline(self):
        cluster_env = ClusterEnv(num_devices=4, mem_budget=4.5*GB)
        model_datas = [
            ModelData("m1", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m4", 1, 5, load_test_prof_result("test-2GB-100ms")),
        ]

        policy = ModelParallelismPlacement()
        group_configs, group_models, _ = policy.solve(model_datas, cluster_env)

        assert len(group_configs) == 2
        assert group_configs[0].pp == 2
        assert group_configs[1].pp == 2
        assert group_models[0] == [0, 1, 2, 3]
        assert group_models[1] == [0, 1, 2, 3]

    def test_placement_api(self):
        for policy in [SelectiveReplication(), SelectiveReplicationWithPipeline()]:
            controller = Controller()
            controller.register_model.remote("m1", EchoModel)
            controller.register_model.remote("m2", EchoModel)
            controller.register_model.remote("m3", EchoModel)
            controller.register_model.remote("m4", EchoModel)

            policy = SelectiveReplication()
            cluster_env = ClusterEnv(num_devices=4, mem_budget=4.5*GB)
            model_datas = [
                ModelData("m1", 1, 5, load_test_prof_result("test-2GB-100ms")),
                ModelData("m2", 1, 5, load_test_prof_result("test-2GB-100ms")),
                ModelData("m3", 1, 5, load_test_prof_result("test-2GB-100ms")),
                ModelData("m4", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ]
            policy.place_models(controller, model_datas, cluster_env)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PlacementPolicyTest("test_selective_replication"))
    suite.addTest(PlacementPolicyTest("test_selective_replication_with_pipeline"))
    suite.addTest(PlacementPolicyTest("test_placement_api"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
