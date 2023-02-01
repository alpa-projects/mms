import argparse
import time

from alpa_serve.simulator.controller import simulate_one_case, approximate_one_case
from alpa_serve.simulator.workload import Workload

from benchmarks.alpa.suite_debug import suite_debug


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="debug_replicate")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bench-speed", action="store_true")
    args = parser.parse_args()

    print("simulate_one_case")
    tic = time.time()
    stats, placement = simulate_one_case(suite_debug[args.case], debug=args.debug)
    print(f"time: {time.time() - tic:.4f}")
    Workload.print_stats(stats)
    print("")

    print("approximate_one_case")
    stats, placement = approximate_one_case(suite_debug[args.case], debug=args.debug)
    tic = time.time()
    stats, placement = approximate_one_case(suite_debug[args.case], debug=args.debug)
    print(f"time: {time.time() - tic:.4f}")
    Workload.print_stats(stats)

    if args.bench_speed:
        tic = time.time()
        stats, placement = approximate_one_case(suite_debug[args.case], debug=args.debug)
        print(f"time: {time.time() - tic:.4f}")
