import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np

from alpasim.cluster import Cluster, Mesh, load_meshexecutors, save_meshexecutors
from alpasim.scheduler import FIFOScheduler
from alpasim.simulator import Simulator
from alpasim.workload import generate_workload, PossoinWorkLoad
from alpasim.utils import compute_statistics_from_simulation, dump_chrome_tracing_from_simulation


def plot_memory_saving(model_size, x1, y1, x2, y2, filename):
    plt.plot(x1, y1, "c", label="selective replication")
    plt.plot(x2, y2, "orange", label="partition + replication")
    plt.plot([model_size, model_size], [0, 0.5], "k--")
    #plt.plot([16, 16], [0, 0.5], "k--")
    #plt.xticks([0, model_size, 16], ["0", f"Model Size({model_size}GB) ", "GPU Memory Capacity(16GB)"])
    plt.xticks([0, model_size], ["0", f"Model Size({model_size}GB) "])
    plt.xlabel("Memory Usage Per GPU (GB)")
    plt.ylabel("99% Tail Latency (s)")
    plt.legend()
    plt.savefig(filename)
 

def run_exp(workload_name, placement_filename, model_id_to_service_name, experiment_name):
    print("\n========================")
    print(f"Run {experiment_name}:")
    workload_filename = f"./workload/{workload_name}"
    workload = PossoinWorkLoad.load(workload_filename)
    cluster = Cluster(1, 16, 16)
    meshexecutors = load_meshexecutors(placement_filename, cluster)
    scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
    simulator = Simulator(scheduler, cluster)
    simulator.start()
    latencies, overall_latencies = compute_statistics_from_simulation(scheduler.completed_requests)
    return np.mean(overall_latencies)
    #dump_chrome_tracing_from_simulation(scheduler.completed_requests, f"./traces/{workload_name}_{experiment_name}.json")


def two_gpu():
    model_size = 2.6
    workload_name = "test_workload_2_even_10Hz_3600s"
    baseline_placement = "./placements/placement_baseline_2GPUs.json"
    pipeline_placement = "./placements/placement_pipeline_2GPUs.json"
    strong_baseline_placement = "./placements/placement_strong_2GPUs.json"
    model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1"}
    l1 = run_exp(workload_name, baseline_placement, model_id_to_service_name, "2 GPU baseline")
    l2 = run_exp(workload_name, pipeline_placement, model_id_to_service_name, "2 GPU pipeline")
    l3 = run_exp(workload_name, strong_baseline_placement, model_id_to_service_name, "2 GPU strong baseline")
    x = [model_size, model_size * 2]
    y1 = [l1, l3]
    y2 = [l2, l3]
    plt.plot(x, y1, "c", label="selective replication")
    plt.plot(x, y2, "orange", label="partition + replication")
    plt.plot([model_size, model_size], [0, 0.5], "k--")
    plt.plot([model_size * 2, model_size * 2], [0, 0.5], "k--")
    plt.text(model_size - 0.7, 0.05, "Model Size", fontsize=12)
    plt.text(model_size * 2 - 0.7, 0.05, "Model Size x #Model", fontsize=12)
    plt.xlim(0, 8)
    plt.ylim(0, 0.5)
    plt.xlabel("Memory Usage Per GPU (GB)")
    plt.ylabel("99% Tail Latency (s)")
    plt.legend()
    plt.title("2 Model - 2 GPU")
    plt.savefig(f"./figures/{workload_name}.png")
 

def four_gpu():
    # workload_name = "test_workload_4_even_20Hz_3600s"
    workload_name = "test_workload_4_even_28Hz_60s"
    pipeline_placement_memx1 = "./placements/placement_pipeline_4GPUs_memx1.json"
    pipeline_placement_memx2 = "./placements/placement_pipeline_4GPUs_memx2.json"
    baseline_placement_memx1 = "./placements/placement_baseline_4GPUs_memx1.json"
    baseline_placement_memx2 = "./placements/placement_baseline_4GPUs_memx2.json"
    baseline_placement_memx4 = "./placements/placement_baseline_4GPUs_memx4.json"
    model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1", 2: "Bert_2.6B_2", 3: "Bert_2.6B_3"}
    #run_exp(workload_name, pipeline_placement_memx1, model_id_to_service_name, "4 GPU pipeline memx1")
    run_exp(workload_name, pipeline_placement_memx2, model_id_to_service_name, "4 GPU pipeline memx2")
    #run_exp(workload_name, baseline_placement_memx1, model_id_to_service_name, "4 GPU baseline memx1")
    #run_exp(workload_name, baseline_placement_memx2, model_id_to_service_name, "4 GPU baseline memx2")
    run_exp(workload_name, baseline_placement_memx4, model_id_to_service_name, "4 GPU baseline memx4")

def eight_gpu():
    workload_name = "test_workload_8_even_40Hz_3600s"
    pipeline_placement_memx1 = "./placements/placement_pipeline_8GPUs_memx1.json"


def workload():
    # generate_workload(2, 10, [0.5]*2, 60, [200]*2, "test_workload_2_even_10Hz_60s")
    # generate_workload(2, 10, [0.5]*2, 3600, [200]*2, "test_workload_2_even_10Hz_3600s")
    # generate_workload(4, 20, [0.25]*4, 3600, [200]*4, "test_workload_4_even_20Hz_3600s")
    # generate_workload(8, 40, [0.125]*8, 3600, [200]*8, "test_workload_8_even_40Hz_3600s")
    # generate_workload(16, 80, [0.0625]*16, 3600, [200]*16, "test_workload_16_even_80Hz_3600s")
    generate_workload(4, 20, [0.25]*4, 60, [200]*4, "test_workload_4_even_20Hz_60s")
    generate_workload(4, 28, [0.25]*4, 60, [200]*4, "test_workload_4_even_28Hz_60s")

if __name__ == "__main__":
    gpu_num = int(sys.argv[1])
    if gpu_num == 2:
        two_gpu()
    elif gpu_num == 4:
        four_gpu()
    elif gpu_num == 8:
        eight_gpu()
    else:
        print("Invalid gpu number")
        exit(1)
