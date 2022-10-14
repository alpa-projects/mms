import argparse

from alpasim.cluster import Cluster, load_meshexecutors
from alpasim.scheduler import FIFOScheduler
from alpasim.utils import dump_chrome_tracing_from_simulation
from alpasim.workload import PossoinWorkLoad
from alpasim.simulator import Simulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster simulator for distributed DL inference tasks")
    parser.add_argument('--name', type=str, required=True,
                        help="simulation name")
    parser.add_argument('-n', '--num_nodes', type=int, default=1,
                        help="number of nodes in the cluster")
    parser.add_argument('-d', '--num_devices_per_node', type=int, default=2,
                        help="number of devices per node in the cluster")
    parser.add_argument('-c', '--memory_capacity', type=int, default=16,
                        help="GPU memory capacity in GB")
    parser.add_argument('-w', '--workload', type=str, required=True,
                        help='Workload Filename')
    parser.add_argument('-p', '--placement', type=str, required=True,
                        help='Placement Filename')
    parser.add_argument('--chrome_trace', action='store_true',
                        help='Dump chrome trace')
    args = parser.parse_args()

    cluster = Cluster(args.num_nodes, args.num_devices_per_node, args.memory_capacity)
    workload = PossoinWorkLoad.load(f"./workload/{args.workload}")
    meshexecutors = load_meshexecutors(f"./placements/{args.placement}", cluster)
    scheduler = FIFOScheduler(workload, meshexecutors)
    simulator = Simulator(scheduler, cluster)
    simulator.start()
    if args.chrome_trace:
        dump_chrome_tracing_from_simulation(scheduler.completed_tasks, f"./chrome_trace/{args.name}.json")
