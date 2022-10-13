"""Test placement policy"""
import unittest

import numpy as np

from alpa_serve.placement_policy import (SelectiveReplication, ModelData,
    SelectiveReplicationWithPipeline)
from alpa.util import GB


class PlacementPolicyTest(unittest.TestCase):

    def test_selective_replication(self):
        policy = SelectiveReplication()

        mem_budget = 2 * GB
        num_gpus = 4
        model_datas = [
            ModelData("m1", 1.0 * GB, 1.0, 1.0),
            ModelData("m2", 1.0 * GB, 1.0, 1.0),
            ModelData("m3", 1.0 * GB, 1.0, 1.0),
            ModelData("m4", 1.0 * GB, 1.0, 1.0),
        ]

        obj, placement = policy.solve(model_datas, num_gpus, mem_budget)

        row_sum = np.sum(placement, axis=1)
        col_sum = np.sum(placement, axis=0)
        assert row_sum.tolist() == [2, 2, 2, 2]
        assert col_sum.tolist() == [2, 2, 2, 2]

    def test_selective_replication_with_pipeline(self):
        mem_budget = 2 * GB
        num_gpus = 4
        model_datas = [
            ModelData("m1", 1.0 * GB, 1.0, 1.0, [(1, 1), (2, 0.95), (4, 0.9)]),
            ModelData("m2", 1.0 * GB, 1.0, 1.0, [(1, 1), (2, 0.95), (4, 0.9)]),
            ModelData("m3", 1.0 * GB, 1.0, 1.0, [(1, 1), (2, 0.95), (4, 0.9)]),
            ModelData("m4", 1.0 * GB, 1.0, 1.0, [(1, 1), (2, 0.95), (4, 0.9)]),
        ]

        policy = SelectiveReplicationWithPipeline()
        obj, (group_sizes, group_models) = policy.solve(
            model_datas, num_gpus, mem_budget, [0, 1, 2, 4])

        assert group_sizes == [2, 2]
        assert group_models == [[0, 1, 2, 3], [0, 1, 2, 3]]


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PlacementPolicyTest("test_selective_replication"))
    suite.addTest(PlacementPolicyTest("test_selective_replication_with_pipeline"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

