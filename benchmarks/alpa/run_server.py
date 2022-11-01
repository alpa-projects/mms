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
    if model == "even":
        controller.register_model.remote("a", BertModel, [bert_specs["1.3B"]])
        controller.register_model.remote("b", BertModel, [bert_specs["1.3B"]])
    elif model == "uneven":
        controller.register_model.remote("a", BertModel, [bert_specs["1.3B"]])
        controller.register_model.remote("b", BertModel, [bert_specs["2.6B"]])
    else:
        raise ValueError(f"Invalid model: {model}")


def place_models(controller, placement):
    if placement == "manual_1":
        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 1])
        controller.create_replica.remote(
            "a", group_id, [ParallelConfig(1, 1, 1)])

        group_id = 1
        controller.create_mesh_group_manager.remote(group_id, [1, 1])
        controller.create_replica.remote(
            "b", group_id, [ParallelConfig(1, 1, 1)])
    elif placement == "manual_2":
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
    elif placement == "manual_3":
        group_id = 0
        controller.create_mesh_group_manager.remote(group_id, [1, 2])
        controller.create_replica.remote(
            "a", group_id, [ParallelConfig(1, 1, 2)])
        controller.create_replica.remote(
            "b", group_id, [ParallelConfig(1, 1, 2)])
    elif placement == "sr":
        raise NotImplementedError()
    elif placement == "srp":
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid placement: {placement}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="even")
    parser.add_argument("--placement", type=str, default="manual_1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    ray.init(address="auto")

    # Launch the controller
    controller = run_controller("localhost", port=args.port, name=None)
    register_models(controller, args.model)
    place_models(controller, args.placement)
    controller.sync()

    print("INIT DONE")
    while True:
        pass
