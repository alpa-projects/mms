"""Test placement policy"""
import unittest

import numpy as np

from alpa_serve.placement_policy import (ModelData, ClusterEnv,
    SelectiveReplication, SelectiveReplicationWithPipeline)
from alpa_serve.profiling import load_test_prof_result
from alpa.util import GB


class PlacementPolicyTest(unittest.TestCase):

    def test_selective_replication(self):
        policy = SelectiveReplication()

        cluster_env = ClusterEnv(num_gpus=4, mem_budget=4.5*GB)
        model_datas = [
            ModelData("m1", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m4", 1, 5, load_test_prof_result("test-2GB-100ms")),
        ]

        obj, placement = policy.solve(model_datas, cluster_env)

        row_sum = np.sum(placement, axis=1)
        col_sum = np.sum(placement, axis=0)
        assert row_sum.tolist() == [2, 2, 2, 2]
        assert col_sum.tolist() == [2, 2, 2, 2]

    def test_selective_replication_with_pipeline(self):
        cluster_env = ClusterEnv(num_gpus=4, mem_budget=4.5*GB)
        model_datas = [
            ModelData("m1", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 1, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m4", 1, 5, load_test_prof_result("test-2GB-100ms")),
        ]

        policy = SelectiveReplicationWithPipeline()
        obj, (group_configs, group_models) = policy.solve(model_datas, cluster_env)

        assert len(group_configs) == 2
        assert group_configs[0].pp == 2
        assert group_configs[1].pp == 2
        assert group_models[0] == [0, 1, 2, 3]
        assert group_models[1] == [0, 1, 2, 3]


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PlacementPolicyTest("test_selective_replication"))
    suite.addTest(PlacementPolicyTest("test_selective_replication_with_pipeline"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
