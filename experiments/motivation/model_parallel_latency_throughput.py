import pickle
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Pdf')
import matplotlib.pyplot as plt
from alpa_serve.util import GB

def get_latency_and_throughput(all_latency):
    latency = sum(all_latency)
    throughput = 1 / max(all_latency)
    return latency, throughput

def plot_database(database, model_name="bert-2.6b"):
    n_gpus = [1, 2, 4, 8]
    pp_latency = []
    pp_throughput = []
    pp_memory = []
    op_latency = []
    op_throughput = []
    op_memory = []
    dp_latency = []
    dp_throughput = []
    dp_memory = []
    for n in n_gpus:
        pp_all_latency = database[model_name].para_dict[(1, 1, n)].latency[1]
        pp_latency.append(get_latency_and_throughput(pp_all_latency)[0])
        pp_throughput.append(get_latency_and_throughput(pp_all_latency)[1])
        pp_memory.append(sum(database[model_name].para_dict[(1, 1, n)].weight_mem) / GB)
        op_all_latency = database[model_name].para_dict[(1, n, 1)].latency[1]
        op_latency.append(get_latency_and_throughput(op_all_latency)[0])
        op_memory.append(sum(database[model_name].para_dict[(1, n, 1)].weight_mem) * n / GB)
        op_throughput.append(get_latency_and_throughput(op_all_latency)[1])
        dp_all_latency = database[model_name].para_dict[(1, 1, 1)].latency[1]
        dp_latency.append(dp_all_latency)
        dp_throughput.append(n / max(dp_all_latency))
        dp_memory.append(sum(database[model_name].para_dict[(1, 1, 1)].weight_mem) * n / GB)

    plt.figure(figsize=(3, 2))
    plt.plot(n_gpus, pp_latency, '.-', label="Inter-op Parallelism")
    plt.plot(n_gpus, op_latency, '.-', label="Intra-op Parallelism")
    plt.plot(n_gpus, dp_latency, '.-', label="Replication")
    # plt.axvline(8, linestyle='--', color = "black", label = "Single Node Boundary", linewidth=0.75)
    plt.xlabel("#GPUs")
    plt.ylim(bottom=0)
    plt.ylabel("Latency (s)")
    plt.grid()
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    plt.savefig(f"model_parallel_latency.pdf")

    plt.figure(figsize=(3, 2))
    plt.plot(n_gpus, pp_throughput, '.-', label="Inter-op Parallelism")
    plt.plot(n_gpus, op_throughput, '.-', label="Intra-op Parallelism")
    plt.plot(n_gpus, dp_throughput, '.-', label="Replication")
    # plt.axvline(8, linestyle='--', color = "black", label = "Single Node Boundary", linewidth=0.75)
    plt.xlabel("#GPUs")
    plt.ylabel("Throughput (req/s)")
    plt.grid()
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    plt.savefig(f"model_parallel_throughput.pdf")

    plt.figure(figsize=(3, 2))
    plt.plot(n_gpus, pp_memory, '.-', label="Inter-op Parallelism")
    plt.plot(n_gpus, op_memory, '.-', label="Intra-op Parallelism")
    plt.plot(n_gpus, dp_memory, '.-', label="Replication")
    # plt.axvline(8, linestyle='--', color = "black", label = "Single Node Boundary", linewidth=0.75)
    plt.xlabel("#GPUs")
    plt.ylabel("Memory (GB)")
    plt.grid()
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    plt.savefig(f"model_parallel_memory.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_filename", type=str, default="profiling_result_long_sequence_manual.pkl")
    args = parser.parse_args()
    with open(args.database_filename, "rb") as f:
        database = pickle.load(f)

    plot_database(database)
