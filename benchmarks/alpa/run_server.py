import argparse
import asyncio

from alpa_serve import run_controller
from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy import (
    SelectiveReplication, SelectiveReplicationWithPipeline,
    ModelData)
from alpa.util import GB
import ray

from benchmarks.alpa.bert_model import BertModel, bert_specs


def register_models(controller, model):
    if model == "tmp":
        controller.register_model.remote("a", BertModel, [bert_specs["1.3B"]])
        controller.register_model.remote("b", BertModel, [bert_specs["1.3B"]])
    else:
        raise ValueError(f"Invalid model: {model}")


def place_models(controller, placement):
    if placement == "manual_1":
        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 1])
        controller.create_replica.remote(
            "a", group_id, [ParallelConfig(1, 1, 1)])
        controller.create_replica.remote(
            "b", group_id, [ParallelConfig(1, 1, 1)])

        group_id = 1
        controller.create_mesh_group_manager.remote(group_id, [1, 1])
        controller.create_replica.remote(
            "a", group_id, [ParallelConfig(1, 1, 1)])
        controller.create_replica.remote(
            "b", group_id, [ParallelConfig(1, 1, 1)])
    elif placement == "manual_2":
        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 2])
        controller.create_replica.remote(
            "a", group_id, [ParallelConfig(1, 1, 2)])
        controller.create_replica.remote(
            "b", group_id, [ParallelConfig(1, 1, 2)])
    elif placement == "sr":
        policy = SelectiveReplication()
        policy.place_models(controller,
            model_datas=[
                ModelData("a", 3*GB, 1, 1),
                ModelData("b", 3*GB, 1, 1),
            ],
            mem_budget=14*GB,
            num_gpus=2,
        )
    elif placement == "srp":
        policy = SelectiveReplicationWithPipeline()
        pipeline_decay = [(1, 1), (2, 0.95), (4, 0.90)]
        policy.place_models(controller, mem_budget=3*GB, num_gpus=2,
            model_datas=[
                ModelData("a", 3*GB, 1, 1, pipeline_decay),
                ModelData("b", 3*GB, 1, 1, pipeline_decay),
            ], group_sizes=(0, 1, 2, 4))
    else:
        raise ValueError(f"Invalid placement: {placement}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tmp")
    parser.add_argument("--placement", type=str, default="manual_1")
    parser.add_argument("--workload", type=str, default="tmp")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    ray.init(address="auto")

    # Launch controller
    controller = run_controller("localhost", port=args.port, name=None)
    register_models(controller, args.model)
    place_models(controller, args.placement)
    controller.sync()

    print("INIT DONE")
    while True:
        pass
