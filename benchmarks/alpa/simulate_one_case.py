import argparse

from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.simulator.controller import Controller, Client
from alpa_serve.simulator.event_loop import run_event_loop
from alpa_serve.simulator.workload import Workload

from benchmarks.alpa.suite import cases
from benchmarks.alpa.run_one_case import run_workload


def simulate_one_case(case, prof_database, debug=False):
    register_models, generate_workload, place_models = case

    # Launch the controller
    controller = Controller()
    register_models(controller, prof_database)
    place_models(controller)

    # Launch the client
    client = Client(controller, debug=debug)
    workload = generate_workload()

    # Run workloads
    return run_event_loop(run_workload(client, workload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_manual_1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    prof_database = ProfilingDatabase("profiling_result.pkl", False)
    stats = simulate_one_case(cases[args.case], prof_database, debug=args.debug)
    Workload.print_stats(stats)
