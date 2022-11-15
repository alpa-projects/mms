import argparse

from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.workload import Workload
from alpa_serve.util import ServingCase

from benchmarks.alpa.suite_debug import suite_debug
from benchmarks.alpa.run_one_case import run_workload


def simulate_one_case(case: ServingCase, debug=False):
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = Controller()
    register_models(controller)
    placement_policy = place_models(controller)

    # Launch the client
    client = Client(controller, debug=debug)
    workload = generate_workload()

    # Run workloads
    stats = run_event_loop(run_workload(client, workload))
    return stats, placement_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_manual_1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    stats, _ = simulate_one_case(suite_debug[args.case], debug=args.debug)
    Workload.print_stats(stats)
