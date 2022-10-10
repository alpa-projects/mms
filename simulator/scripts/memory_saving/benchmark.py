import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np

from alpasim.cluster import Cluster, Mesh, load_meshexecutors, save_meshexecutors
from alpasim.scheduler import FIFOScheduler
from alpasim.simulator import Simulator
from alpasim.workload import generate_workload, PossoinWorkLoad
from alpasim.utils import compute_statistics_from_simulation, dump_chrome_tracing_from_simulation


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


def two_gpu():
    #workload_name = "test_workload_2_even_10Hz_60s"
    workload_name = "test_workload_2_even_10Hz_3600s"
    baseline_placement = "./placements/placement_baseline_2GPUs.json"
    pipeline_placement = "./placements/placement_pipeline_2GPUs.json"
    strong_baseline_placement = "./placements/placement_strong_2GPUs.json"
    model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1"}
    run_exp(workload_name, baseline_placement, model_id_to_service_name, "2 GPU baseline")
    run_exp(workload_name, pipeline_placement, model_id_to_service_name, "2 GPU pipeline")
    run_exp(workload_name, strong_baseline_placement, model_id_to_service_name, "2 GPU strong baseline")

def four_gpu():
    workload_name = "test_workload_4_even_20Hz_3600s"
    pipeline_placement_memx1 = "./placements/placement_pipeline_4GPUs_memx1.json"
    pipeline_placement_memx2 = "./placements/placement_pipeline_4GPUs_memx2.json"
    baseline_placement_memx1 = "./placements/placement_baseline_4GPUs_memx1.json"
    baseline_placement_memx2 = "./placements/placement_baseline_4GPUs_memx2.json"
    baseline_placement_memx4 = "./placements/placement_baseline_4GPUs_memx4.json"
    model_id_to_service_name = {0: "Bert_2.6B_0", 1: "Bert_2.6B_1", 2: "Bert_2.6B_2", 3: "Bert_2.6B_3"}
    run_exp(workload_name, pipeline_placement_memx1, model_id_to_service_name, "4 GPU pipeline memx1")
    run_exp(workload_name, pipeline_placement_memx2, model_id_to_service_name, "4 GPU pipeline memx2")
    run_exp(workload_name, baseline_placement_memx1, model_id_to_service_name, "4 GPU baseline memx1")
    run_exp(workload_name, baseline_placement_memx2, model_id_to_service_name, "4 GPU baseline memx2")
    run_exp(workload_name, baseline_placement_memx4, model_id_to_service_name, "4 GPU baseline memx4")

def eight_gpu():
    pass


def workload():
    generate_workload(2, 10, [0.5]*2, 60, [200]*2, "test_workload_2_even_10Hz_60s")
    generate_workload(2, 10, [0.5]*2, 3600, [200]*2, "test_workload_2_even_10Hz_3600s")
    generate_workload(4, 20, [0.25]*4, 3600, [200]*4, "test_workload_4_even_20Hz_3600s")
    generate_workload(8, 40, [0.125]*8, 3600, [200]*8, "test_workload_8_even_40Hz_3600s")
    generate_workload(16, 80, [0.0625]*16, 3600, [200]*16, "test_workload_16_even_80Hz_3600s")

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
