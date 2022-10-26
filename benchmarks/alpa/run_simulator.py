import argparse
from functools import partial

from alpa_serve.profiling import ProfilingResult, ParallelConfig
from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.executable import Executable

from benchmarks.alpa.run_server import place_models
from benchmarks.alpa.run_client import generate_workload, run_workload


def register_models(controller, model):
    if model == "tmp":
        controller.register_model.remote(
            "a", partial(Executable, ProfilingResult.load("alpa/bert-1.3b")))
        controller.register_model.remote(
            "b", partial(Executable, ProfilingResult.load("alpa/bert-1.3b")))
    else:
        raise ValueError(f"Invalid model: {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tmp")
    parser.add_argument("--placement", type=str, default="manual_1")
    parser.add_argument("--workload", type=str, default="tmp")
    args = parser.parse_args()

    # Launch controller
    controller = Controller()
    register_models(controller, args.model)
    place_models(controller, args.placement)
    controller.sync()

    # Launch client
    client = Client(controller)
    workload = generate_workload(args.workload)

    # Run workloads
    run_event_loop(run_workload(client, workload))
