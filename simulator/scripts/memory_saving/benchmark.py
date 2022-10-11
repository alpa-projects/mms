import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np

from alpasim.cluster import Cluster, Mesh, load_meshexecutors, save_meshexecutors
from alpasim.scheduler import FIFOScheduler
from alpasim.simulator import Simulator
from alpasim.workload import generate_workload, PossoinWorkLoad
from alpasim.utils import compute_statistics_from_simulation, dump_chrome_tracing_from_simulation


def plot_memory_saving(model_size, model_nums, x1, y1, x2, y2, xlim, ylim, ylabel, title, filename):
    plt.figure()
    plt.plot(x1, y1, "c", label="selective replication")
    plt.plot(x2, y2, "orange", label="partition + replication")
    plt.plot([model_size, model_size], [0, 5], "k--")
    plt.plot([model_size * model_nums, model_size * model_nums], [0, 5], "k--")
    plt.text(model_size - 0.7, 0.05, "Model Size", fontsize=8)
    plt.text(model_size * model_nums - 0.7, 0.05, "Model Size x #Model", fontsize=8)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Memory Usage Per GPU (GB)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
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
    sorted_latencies = np.sort(overall_latencies)
    return np.mean(overall_latencies), sorted_latencies[int(0.99 * len(sorted_latencies))]
    #dump_chrome_tracing_from_simulation(scheduler.completed_requests, f"./traces/{workload_name}_{experiment_name}.json")


def two_gpu():
    model_size = 2.6
    workload_name = "test_workload_2_even_10Hz_3600s"
    #workload_name = "test_workload_2_even_12Hz_3600s"
    #workload_name = "test_workload_2_even_14Hz_3600s"
    baseline_placement = "./placements/placement_baseline_2GPUs.json"
    pipeline_placement = "./placements/placement_pipeline_2GPUs.json"
    strong_baseline_placement = "./placements/placement_strong_2GPUs.json"
    model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1"}
    l1_m, l1_t = run_exp(workload_name, baseline_placement, model_id_to_service_name, "2 GPU baseline")
    l2_m, l2_t = run_exp(workload_name, pipeline_placement, model_id_to_service_name, "2 GPU pipeline")
    l3_m, l3_t = run_exp(workload_name, strong_baseline_placement, model_id_to_service_name, "2 GPU strong baseline")
    x = [model_size, model_size * 2]
    y1_m, y2_m = [l1_m, l3_m], [l2_m, l3_m]
    lim_m = max(l1_m, l2_m, l3_m) + 0.2
    y1_t, y2_t = [l1_t, l3_t], [l2_t, l3_t]
    lim_t = max(l1_t, l2_t, l3_t) + 0.2
    plot_memory_saving(model_size, 2, x, y1_m, x, y2_m, (0, 8), (0, lim_m), "Mean Latency (s)", workload_name, f"./figures/{workload_name}_mean.png")
    plot_memory_saving(model_size, 2, x, y1_t, x, y2_t, (0, 8), (0, lim_t), "99% Tail Latency (s)", workload_name, f"./figures/{workload_name}_tail.png")


def four_gpu():
    model_size = 2.6
    # workload_name = "test_workload_4_even_20Hz_3600s"
    # workload_name = "test_workload_4_even_22Hz_3600s"
    # workload_name = "test_workload_4_even_24Hz_3600s"
    workload_name = "test_workload_4_even_26Hz_3600s"
    # workload_name = "test_workload_4_even_28Hz_3600s"
    pipeline_placement_memx1 = "./placements/placement_pipeline_4GPUs_memx1.json"
    pipeline_placement_memx2 = "./placements/placement_pipeline_4GPUs_memx2.json"
    baseline_placement_memx1 = "./placements/placement_baseline_4GPUs_memx1.json"
    baseline_placement_memx2 = "./placements/placement_baseline_4GPUs_memx2.json"
    baseline_placement_memx4 = "./placements/placement_baseline_4GPUs_memx4.json"
    model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1", 2: "Bert_2.6B_2", 3: "Bert_2.6B_3"}
    pl1_m, pl1_t = run_exp(workload_name, pipeline_placement_memx1, model_id_to_service_name, "4 GPU pipeline memx1")
    pl2_m, pl2_t = run_exp(workload_name, pipeline_placement_memx2, model_id_to_service_name, "4 GPU pipeline memx2")
    bl1_m, bl1_t = run_exp(workload_name, baseline_placement_memx1, model_id_to_service_name, "4 GPU baseline memx1")
    bl2_m, bl2_t = run_exp(workload_name, baseline_placement_memx2, model_id_to_service_name, "4 GPU baseline memx2")
    bl4_m, bl4_t = run_exp(workload_name, baseline_placement_memx4, model_id_to_service_name, "4 GPU baseline memx4")
    x = [model_size, model_size * 2, model_size * 4]
    y1_m, y2_m = [pl1_m, pl2_m, bl4_m], [bl1_m, bl2_m, bl4_m]
    lim_m = max(pl1_m, pl2_m, bl1_m, bl2_m, bl4_m) + 0.2
    y1_t, y2_t = [pl1_t, pl2_t, bl4_t], [bl1_t, bl2_t, bl4_t]
    lim_t = max(pl1_t, pl2_t, bl1_t, bl2_t, bl4_t) + 0.2
    plot_memory_saving(model_size, 4, x, y1_m, x, y2_m, (0, 12), (0, lim_m), "Mean Latency (s)", workload_name, f"./figures/{workload_name}_mean.png")
    plot_memory_saving(model_size, 4, x, y1_t, x, y2_t, (0, 12), (0, lim_t), "99% Tail Latency (s)", workload_name, f"./figures/{workload_name}_tail.png")


def eight_gpu():
    workload_name = "test_workload_8_even_40Hz_3600s"
    pipeline_placement_memx1 = "./placements/placement_pipeline_8GPUs_memx1.json"


def workload():
    # generate_workload(2, 10, [0.5]*2, 60, [200]*2, "./workload/test_workload_2_even_10Hz_60s")
    # generate_workload(2, 10, [0.5]*2, 3600, [200]*2, "./workload/test_workload_2_even_10Hz_3600s")
    # generate_workload(2, 12, [0.5]*2, 3600, [200]*2, "./workload/test_workload_2_even_12Hz_3600s")
    # generate_workload(2, 14, [0.5]*2, 3600, [200]*2, "./workload/test_workload_2_even_14Hz_3600s")

    generate_workload(4, 20, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_20Hz_3600s")
    generate_workload(4, 22, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_22Hz_3600s")
    generate_workload(4, 24, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_24Hz_3600s")
    generate_workload(4, 26, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_26Hz_3600s")
    generate_workload(4, 28, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_28Hz_3600s")

    # generate_workload(8, 40, [0.125]*8, 3600, [200]*8, "test_workload_8_even_40Hz_3600s")
    # generate_workload(16, 80, [0.0625]*16, 3600, [200]*16, "test_workload_16_even_80Hz_3600s")

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
