import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import numpy as np

from alpasim.cluster import Cluster, Mesh, load_meshexecutors, save_meshexecutors
from alpasim.scheduler import FIFOScheduler
from alpasim.simulator import Simulator
from alpasim.workload import generate_workload, PossoinWorkLoad
from alpasim.utils import compute_statistics_from_cluster_trace, compute_statistics_from_simulation, \
                          dump_chrome_tracing_from_simulation, dump_chrome_tracing_from_cluster_trace

def plot_cdf(latencies, overall_latencies, is_baseline):
    model0_latencies = latencies[0]
    model1_latencies = latencies[1]
    # sort data
    x1, x2, x = np.sort(model0_latencies), np.sort(model1_latencies), np.sort(overall_latencies)
    # calculate CDF values
    y1, y2, y = 1. * np.arange(len(model0_latencies)) / (len(model0_latencies) - 1), \
                1. * np.arange(len(model1_latencies)) / (len(model1_latencies) - 1), \
                1. * np.arange(len(overall_latencies)) / (len(overall_latencies) - 1),
    # plot CDF
    if is_baseline:
        plt.plot(x1, y1, ":", color="c", label="baseline model0")
        plt.plot(x2, y2, "-.", color="c", label="baseline model1")
        plt.plot(x, y, "-", color="c", label="baseline overall")
    else:
        plt.plot(x1, y1, ":", color="orange", label="parallel model0")
        plt.plot(x2, y2, "-.", color="orange", label="parallel model1")
        plt.plot(x, y, "-", color="orange", label="parallel overall")


def run_strong_baseline():
    workload_name = "test_workload_8to2_50Hz_60s"
    placement_filename = "./placements/placement_125M_baseline.json"
    model_id_to_service_name = {0: "Bert_125M_0", 1: "Bert_125M_1"}
    print("\n========================")
    print("Test 125M baseline trace:")
    workload_filename = f"./workload/{workload_name}"
    workload = PossoinWorkLoad.load(workload_filename)
    cluster = Cluster(1, 2, 16)
    meshexecutors = load_meshexecutors(placement_filename, cluster)
    scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
    simulator = Simulator(scheduler, cluster)
    simulator.start()
    latencies, overall_latencies = compute_statistics_from_simulation(scheduler.completed_tasks)
    plot_cdf(latencies, overall_latencies, True)

def run_interop():
    workload_name = "test_workload_8to2_50Hz_60s"
    placement_filename = "./placements/placement_125M_interop.json"
    model_id_to_service_name = {0: "Bert_125M_0", 1: "Bert_125M_1"}
    print("\n========================")
    print("Test 125M interop trace:")
    workload_filename = f"./workload/{workload_name}"
    workload = PossoinWorkLoad.load(workload_filename)
    cluster = Cluster(1, 2, 16)
    meshexecutors = load_meshexecutors(placement_filename, cluster)
    scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
    simulator = Simulator(scheduler, cluster)
    simulator.start()
    latencies, overall_latencies = compute_statistics_from_simulation(scheduler.completed_tasks)
    plot_cdf(latencies, overall_latencies, False)

def run_intraop():
    workload_name = "test_workload_8to2_50Hz_60s"
    placement_filename = "./placements/placement_125M_intraop.json"
    model_id_to_service_name = {0: "Bert_125M_0", 1: "Bert_125M_1"}
    print("\n========================")
    print("Test 125M intraop trace:")
    workload_filename = f"./workload/{workload_name}"
    workload = PossoinWorkLoad.load(workload_filename)
    cluster = Cluster(1, 2, 16)
    meshexecutors = load_meshexecutors(placement_filename, cluster)
    scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
    simulator = Simulator(scheduler, cluster)
    simulator.start()
    latencies, overall_latencies = compute_statistics_from_simulation(scheduler.completed_tasks)
    plot_cdf(latencies, overall_latencies, False)

parallel_method = "interop"
#parallel_method = "intraop"

plt.figure()
run_strong_baseline()
if parallel_method == "interop":
    run_interop()
else:
    run_intraop()

plt.legend()
plt.ylabel("CDF")
plt.xlabel("Latency(s)")

# savefig
plt.savefig(f"cdf_{parallel_method}")






