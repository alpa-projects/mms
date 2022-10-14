import argparse
import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np

from alpasim.cluster import Cluster, load_meshexecutors
from alpasim.scheduler import FIFOScheduler
from alpasim.simulator import Simulator
from alpasim.utils import compute_statistics_from_simulation, dump_chrome_tracing_from_simulation
from alpasim.workload import generate_workload, PossoinWorkLoad


def plot_memory_saving(model_size, model_nums, x1_m, y1_m, x2_m, y2_m, x1_t, y1_t, x2_t, y2_t, xlim, ylim, title, filename):
    plt.figure()
    plt.plot(x1_t, y1_t, "b--", label="selective replication (99% tail latency)")
    plt.plot(x2_t, y2_t,  "r--", label="selective replication + pipeline (99% tail latency)")
    plt.plot(x1_m, y1_m, "b", label="selective replication (mean latency)")
    plt.plot(x2_m, y2_m, "r", label="selective replication + pipeline (mean latency)")
    plt.plot([model_size, model_size], [0, 5], "k--")
    plt.plot([model_size * model_nums, model_size * model_nums], [0, 5], "k--")
    plt.text(model_size - 0.7, 0.05, "Model Size", fontsize=8)
    plt.text(model_size * model_nums - 0.7, 0.05, "Model Size x #Model", fontsize=8)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Memory Usage Per GPU (GB)")
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
 

def run_exp(workload_name, placement_filename, model_id_to_service_name, experiment_name, dump_trace=False):
    print("\n========================")
    print(f"Run {experiment_name}:")
    workload_filename = f"./workload/{workload_name}"
    workload = PossoinWorkLoad.load(workload_filename)
    cluster = Cluster(1, 16, 16)
    meshexecutors = load_meshexecutors(placement_filename, cluster)
    scheduler = FIFOScheduler(workload, meshexecutors, model_id_to_service_name)
    simulator = Simulator(scheduler, cluster)
    simulator.start()
    latencies, overall_latencies = compute_statistics_from_simulation(scheduler.completed_tasks)
    sorted_latencies = np.sort(overall_latencies)
    if dump_trace:
        dump_chrome_tracing_from_simulation(scheduler.completed_tasks, f"./traces/{workload_name}_{experiment_name}.json")
    return np.mean(overall_latencies), sorted_latencies[int(0.99 * len(sorted_latencies))]


def two_gpu():
    model_size = 2.6
    throughput = 10
    workload_name = f"test_workload_2_even_{throughput}Hz_3600s"
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
    plot_memory_saving(model_size, 2, x, y1_m, x, y2_m, x, y1_t, x, y2_t, (0, 8), (0, lim_t), workload_name, f"./figures/{workload_name}.png")


def four_gpu():
    model_size = 2.6
    throughput = 26
    workload_name = f"test_workload_4_even_{throughput}Hz_3600s"
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
    y1_m, y2_m = [bl1_m, bl2_m, bl4_m], [pl1_m, pl2_m, bl4_m]
    lim_m = max(pl1_m, pl2_m, bl1_m, bl2_m, bl4_m) + 0.2
    y1_t, y2_t = [bl1_t, bl2_t, bl4_t], [pl1_t, pl2_t, bl4_t]
    lim_t = max(pl1_t, pl2_t, bl1_t, bl2_t, bl4_t) + 0.2
    plot_memory_saving(model_size, 4, x, y1_m, x, y2_m, x, y1_t, x, y2_t, (0, 12), (0, lim_t), workload_name, f"./figures/{workload_name}.png")


def four_gpu_uneven():
    model_size = 2.6
    throughput=16
    workload_name = f"test_workload_4_uneven3to1_{throughput}Hz_3600s"
    baseline_placement_memx1 = "./placements/placement_baseline_4GPUs_memx1.json"
    baseline_placement_memx2 = "./placements/placement_baseline_4GPUs_memx2_3to1.json"
    baseline_placement_memx3 = "./placements/placement_baseline_4GPUs_memx3_3to1.json"
    baseline_placement_memx4 = "./placements/placement_baseline_4GPUs_memx4.json"
    pipeline_placement_memx1 = "./placements/placement_pipeline_4GPUs_memx1.json"
    pipeline_placement_memx2 = "./placements/placement_pipeline_4GPUs_memx2.json"
    model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1", 2: "Bert_2.6B_2", 3: "Bert_2.6B_3"}
    bl1_m, bl1_t = run_exp(workload_name, baseline_placement_memx1, model_id_to_service_name, "4 GPU baseline memx1")
    bl2_m, bl2_t = run_exp(workload_name, baseline_placement_memx2, model_id_to_service_name, "4 GPU baseline memx2")
    bl3_m, bl3_t = run_exp(workload_name, baseline_placement_memx3, model_id_to_service_name, "4 GPU baseline memx3")
    bl4_m, bl4_t = run_exp(workload_name, baseline_placement_memx4, model_id_to_service_name, "4 GPU baseline memx4")
    pl1_m, pl1_t = run_exp(workload_name, pipeline_placement_memx1, model_id_to_service_name, "4 GPU pipeline memx1")
    pl2_m, pl2_t = run_exp(workload_name, pipeline_placement_memx2, model_id_to_service_name, "4 GPU pipeline memx2")
    x1 = [model_size, model_size * 2, model_size * 3, model_size * 4]
    y1_m, y2_m = [bl1_m, bl2_m, bl3_m, bl4_m], [pl1_m, pl2_m, bl4_m]
    lim_m = max(pl1_m, pl2_m, bl1_m, bl2_m, bl4_m) + 0.2
    x2 = [model_size, model_size * 2, model_size * 4]
    y1_t, y2_t = [bl1_t, bl2_t, bl3_t, bl4_t], [pl1_t, pl2_t, bl4_t]
    lim_t = max(pl1_t, pl2_t, bl1_t, bl2_t, bl4_t) + 0.2
    plot_memory_saving(model_size, 4, x1, y1_m, x2, y2_m, x1, y1_t, x2, y2_t, (0, 12), (0, lim_t), workload_name, f"./figures/{workload_name}.png")


def eight_gpu():
    workload_name = "test_workload_8_even_40Hz_3600s"
    pipeline_placement_memx1 = "./placements/placement_pipeline_8GPUs_memx1.json"


def workload():
    # 2GPU workload
    generate_workload(2, 10, [0.5]*2, 3600, [200]*2, "./workload/test_workload_2_even_10Hz_3600s")
    generate_workload(2, 12, [0.5]*2, 3600, [200]*2, "./workload/test_workload_2_even_12Hz_3600s")
    generate_workload(2, 14, [0.5]*2, 3600, [200]*2, "./workload/test_workload_2_even_14Hz_3600s")

    # 4GPU even workload
    generate_workload(4, 20, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_20Hz_3600s")
    generate_workload(4, 22, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_22Hz_3600s")
    generate_workload(4, 24, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_24Hz_3600s")
    generate_workload(4, 26, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_26Hz_3600s")
    generate_workload(4, 28, [0.25]*4, 3600, [200]*4, "./workload/test_workload_4_even_28Hz_3600s")

    # 4GPU uneven workload
    generate_workload(4, 16, [3/8, 3/8, 1/8, 1/8], 3600, [200]*4, "./workload/test_workload_4_uneven3to1_16Hz_3600s")
    generate_workload(4, 17, [3/8, 3/8, 1/8, 1/8], 3600, [200]*4, "./workload/test_workload_4_uneven3to1_17Hz_3600s")

    # 8GPU workload
    generate_workload(8, 40, [0.125]*8, 3600, [200]*8, "test_workload_8_even_40Hz_3600s")
    generate_workload(16, 80, [0.0625]*16, 3600, [200]*16, "test_workload_16_even_80Hz_3600s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory saving experiments")
    parser.add_argument('--gpu_num', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--workload', default=False, action='store_true', help='Generate new workload')
    args = parser.parse_args()
    if args.workload:
        workload()

    if args.gpu_num == 2:
        two_gpu()
    elif args.gpu_num == 4:
        four_gpu_uneven()
    elif args.gpu_num == 8:
        eight_gpu()
    else:
        print("Invalid gpu number")
        exit(1)
