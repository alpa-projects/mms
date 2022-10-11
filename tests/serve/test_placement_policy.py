"""Test placement policy"""
import unittest

import numpy as np

from alpa_serve.placement_policy import (SelectiveReplication, ModelData)


class PlacementPolicyTest(unittest.TestCase):

    def test_selective_replication(self):
        policy = SelectiveReplication()

        mem_budget = 2
        num_gpus = 4
        model_datas = [
            ModelData("m1", 1.0, 1.0, 1.0),
            ModelData("m2", 1.0, 1.0, 1.0),
            ModelData("m3", 1.0, 1.0, 1.0),
            ModelData("m4", 1.0, 1.0, 1.0),
        ]

        placement = policy.solve(mem_budget, num_gpus, model_datas)

        row_sum = np.sum(placement, axis=1)
        col_sum = np.sum(placement, axis=0)
        assert row_sum.tolist() == [2, 2, 2, 2]
        assert col_sum.tolist() == [2, 2, 2, 2]


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PlacementPolicyTest("test_selective_replication"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

