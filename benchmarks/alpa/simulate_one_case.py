import argparse

from alpa_serve.simulator.controller import simulate_one_case
from alpa_serve.simulator.workload import Workload

from benchmarks.alpa.suite_debug import suite_debug


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_replicate")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    stats, placement = simulate_one_case(suite_debug[args.case], debug=args.debug)
    Workload.print_stats(stats)
