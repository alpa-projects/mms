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
            ModelData("m1", 2, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 2, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 2, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m4", 2, 5, load_test_prof_result("test-2GB-100ms")),
        ]

        obj, placement = policy.solve(model_datas, cluster_env)

        row_sum = np.sum(placement, axis=1)
        col_sum = np.sum(placement, axis=0)
        assert row_sum.tolist() == [2, 2, 2, 2]
        assert col_sum.tolist() == [2, 2, 2, 2]

    def test_selective_replication_with_pipeline(self):
        cluster_env = ClusterEnv(num_gpus=4, mem_budget=4*GB)
        model_datas = [
            ModelData("m1", 2, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m2", 2, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m3", 2, 5, load_test_prof_result("test-2GB-100ms")),
            ModelData("m4", 2, 5, load_test_prof_result("test-2GB-100ms")),
        ]

        policy = SelectiveReplicationWithPipeline()
        obj, (group_configs, group_models) = policy.solve(model_datas, cluster_env)

        print(group_configs)
        print(group_models)


def suite():
    suite = unittest.TestSuite()
    #suite.addTest(PlacementPolicyTest("test_selective_replication"))
    suite.addTest(PlacementPolicyTest("test_selective_replication_with_pipeline"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
