import argparse

from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.workload import Workload

from benchmarks.alpa.suite import cases
from benchmarks.alpa.run_one_case import run_workload


def simulate_one_case(case):
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = Controller()
    register_models(controller)
    place_models(controller)

    # Launch the client
    client = Client(controller)
    workload = generate_workload()

    # Run workloads
    return run_event_loop(run_workload(client, workload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_manual_1")
    args = parser.parse_args()

    stats = simulate_one_case(cases[args.case])
    Workload.print_stats(stats)
