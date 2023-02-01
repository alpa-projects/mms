"""The model placement policy"""

from alpa_serve.placement_policy.base_policy import ModelData, ClusterEnv
from alpa_serve.placement_policy.model_parallelism import (
    ModelParallelismILP, ModelParallelismRR,
    ModelParallelismGreedy, ModelParallelismSearch,
    ModelParallelismEqual)
from alpa_serve.placement_policy.selective_replication import (
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    SelectiveReplicationUniform, SelectiveReplicationSearch,
    SelectiveReplicationReplacement)
