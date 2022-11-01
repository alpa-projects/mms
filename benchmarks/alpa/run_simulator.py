import argparse
from functools import partial

from alpa_serve.profiling import ParallelConfig, load_test_prof_result
from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.executable import Executable

from benchmarks.alpa.run_server import place_models
from benchmarks.alpa.run_client import generate_workload, run_workload


def register_models(controller, model):
    if model == "even":
        controller.register_model.remote(
            "a", partial(Executable, load_test_prof_result("alpa/bert-1.3b")))
        controller.register_model.remote(
            "b", partial(Executable, load_test_prof_result("alpa/bert-1.3b")))
    elif model == "uneven":
        controller.register_model.remote(
            "a", partial(Executable, load_test_prof_result("alpa/bert-1.3b")))
        controller.register_model.remote(
            "b", partial(Executable, load_test_prof_result("alpa/bert-2.6b")))
    else:
        raise ValueError(f"Invalid model: {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="even")
    parser.add_argument("--placement", type=str, default="manual_1")
    parser.add_argument("--workload", type=str, default="tmp")
    args = parser.parse_args()

    # Launch the controller
    controller = Controller()
    register_models(controller, args.model)
    place_models(controller, args.placement)
    controller.sync()

    # Launch the client
    client = Client(controller)
    workload = generate_workload(args.workload)

    # Run workloads
    run_event_loop(run_workload(client, workload))
